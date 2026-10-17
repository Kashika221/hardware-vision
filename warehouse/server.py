"""FastAPI server — dashboard, REST endpoints, live pose over WebSocket."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import database, logging_setup, metrics, paper_map, vision, workflows

logging_setup.install()
from .grid import build_static_grid, render_dynamic, world_to_cell
from .models import RobotState, Warehouse
from .robot import Robot


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "data"
REF_DIR = DATA_DIR / "refs"
VID_DIR = DATA_DIR / "videos"
REF_DIR.mkdir(parents=True, exist_ok=True)
VID_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Autonomous Inventory Robot")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ─── global state ────────────────────────────────────────────────────────────

_ROBOT: Robot | None = None
_CLIENTS: set[WebSocket] = set()
_LOOP: asyncio.AbstractEventLoop | None = None


def _ensure_robot() -> Robot:
    global _ROBOT
    db = database.load()
    if db.warehouse is None:
        raise HTTPException(400, "no warehouse loaded — import a paper map first")
    if _ROBOT is None or _ROBOT.wh != db.warehouse:
        _ROBOT = Robot(db.warehouse)
        _ROBOT.on_update(_broadcast)
    return _ROBOT


async def _broadcast(state: RobotState):
    dead = []
    payload = json.dumps({"type": "pose", "state": state.model_dump()})
    for ws in list(_CLIENTS):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _CLIENTS.discard(ws)


# ─── UI ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page():
    return (WEB_DIR / "metrics.html").read_text(encoding="utf-8")


# ─── warehouse / map ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    db = database.load()
    if db.products:
        try:
            vision.rehydrate_from_products(db.products)
            print(f"[startup] YOLOE prompts rehydrated: {vision.locked_names()}")
        except Exception as e:
            print(f"[warn] YOLOE rehydrate failed: {e}")


@app.get("/api/state")
async def get_state():
    db = database.load()
    robot_state = _ROBOT.state.model_dump() if _ROBOT else None
    return {
        "warehouse": db.warehouse.model_dump() if db.warehouse else None,
        "products": [p.model_dump() for p in db.products],
        "robot": robot_state,
        "locked": vision.locked_names(),
    }


@app.get("/api/grid")
async def get_grid():
    db = database.load()
    if db.warehouse is None:
        raise HTTPException(400, "no warehouse loaded")
    static = build_static_grid(db.warehouse)
    if _ROBOT is not None:
        r, c = world_to_cell(db.warehouse, _ROBOT.state.x_m, _ROBOT.state.y_m)
        g = render_dynamic(static, (r, c))
    else:
        g = static
    return {
        "rows": db.warehouse.rows,
        "cols": db.warehouse.cols,
        "cell_size_m": db.warehouse.cell_size_m,
        "cells": g.tolist(),
    }


@app.post("/api/map/synth")
async def synth_map(
    width_m: float = Form(10.0),
    height_m: float = Form(6.0),
    n_shelves: int = Form(4),
    seed: Optional[int] = Form(None),
):
    im = paper_map.synthesize_paper_map(width_m, height_m, n_shelves, seed=seed)
    return Response(content=paper_map.pil_to_bytes(im, "PNG"), media_type="image/png")


@app.post("/api/map/import")
async def import_map(
    image: UploadFile = File(...),
    width_m: float = Form(...),
    height_m: float = Form(...),
    cell_size_m: float = Form(0.10),
):
    data = await image.read()
    wh, warped = paper_map.import_paper_map(data, width_m, height_m, cell_size_m)
    # Fresh map ⇒ fresh inventory + fresh vision prompts + fresh metrics
    database.set_warehouse(wh, reset_inventory=True)
    vision.clear_prompts()
    metrics.reset_all()

    global _ROBOT
    _ROBOT = Robot(wh)
    _ROBOT.on_update(_broadcast)

    _, buf = cv2.imencode(".png", warped)
    warped_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "warehouse": wh.model_dump(),
        "warped_png_b64": warped_b64,
    }


@app.post("/api/map/rename_shelf")
async def rename_shelf(old_id: str = Form(...), new_id: str = Form(...)):
    db = database.load()
    if db.warehouse is None:
        raise HTTPException(400, "no warehouse")
    for s in db.warehouse.shelves:
        if s.id == old_id:
            s.id = new_id
            s.qr = f"QR-{new_id}"
    for p in db.products:
        if p.shelf_id == old_id:
            p.shelf_id = new_id
    database.save(db)
    return {"ok": True}


@app.get("/api/shelf/{shelf_id}/qr.png")
async def shelf_qr(shelf_id: str):
    shelf = database.get_shelf(shelf_id)
    if shelf is None:
        raise HTTPException(404, "shelf not found")
    return Response(content=vision.make_qr_png(shelf.qr), media_type="image/png")


# ─── stock / pick ────────────────────────────────────────────────────────────

@app.post("/api/stock")
async def stock_endpoint(
    name: str = Form(...),
    barcode: Optional[str] = Form(None),
    ref_image: Optional[UploadFile] = File(None),
    x1: Optional[int] = Form(None),
    y1: Optional[int] = Form(None),
    x2: Optional[int] = Form(None),
    y2: Optional[int] = Form(None),
    shelf_id: Optional[str] = Form(None),
):
    robot = _ensure_robot()
    ref_path: Optional[str] = None
    bbox: Optional[list[int]] = None
    vpe_error: Optional[str] = None
    if ref_image is not None:
        raw = await ref_image.read()
        p = REF_DIR / f"{name.lower().replace(' ', '_')}.png"
        p.write_bytes(raw)
        ref_path = str(p)
        if None not in (x1, y1, x2, y2):
            bbox = [int(x1), int(y1), int(x2), int(y2)]
        try:
            vision.add_product_prompt(name, ref_path, bbox)
            if name.lower() not in [n.lower() for n in vision.locked_names()]:
                detail = vision.last_vpe_error()
                vpe_error = f"YOLOE did not register the class ({detail or 'unknown'})"
        except Exception as e:
            import traceback
            traceback.print_exc()
            vpe_error = f"{type(e).__name__}: {e}"
    shelf_id_clean = shelf_id.strip() if shelf_id else None
    result = await workflows.stock(robot, name, barcode, ref_path, bbox,
                                   shelf_id=shelf_id_clean or None)
    if vpe_error:
        result["vpe_error"] = vpe_error
    result["locked"] = vision.locked_names()
    return JSONResponse(result)


@app.post("/api/pick")
async def pick_endpoint(name: str = Form(...)):
    robot = _ensure_robot()
    result = await workflows.pick(robot, name)
    return JSONResponse(result)


@app.post("/api/detect_at_shelf")
async def detect_at_shelf(image: UploadFile = File(...)):
    """Upload a photo of the shelf — YOLOE searches for the locked product."""
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "cannot decode image")
    detections = vision.detect_product(bgr)
    ann = vision.annotated(bgr, detections)
    _, buf = cv2.imencode(".jpg", ann)
    return {
        "count": len(detections),
        "detections": detections,
        "annotated_jpg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
    }


# ─── logs ────────────────────────────────────────────────────────────────────

@app.get("/api/logs")
async def get_logs(n: int = 300):
    return {
        "path": str(logging_setup.LOG_PATH),
        "tail": logging_setup.tail(n),
    }


# ─── metrics ─────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
async def get_metrics():
    db = database.load()
    snap = metrics.snapshot(warehouse=db.warehouse)
    snap["shelves"] = [s.model_dump() for s in (db.warehouse.shelves if db.warehouse else [])]
    snap["products"] = [p.model_dump() for p in db.products]
    snap["warehouse"] = db.warehouse.model_dump() if db.warehouse else None
    snap["locked"] = vision.locked_names()
    return snap


# ─── LIVE YOLOE DETECTION — webcam / video MJPEG streams ────────────────────

_STREAM_STOP = threading.Event()


@app.post("/api/upload_video")
async def upload_video(video: UploadFile = File(...)):
    dest = VID_DIR / video.filename
    dest.write_bytes(await video.read())
    return {"ok": True, "path": str(dest)}


@app.post("/api/stream/stop")
async def stop_stream():
    _STREAM_STOP.set()
    return {"ok": True}


def _mjpeg_generator(cap: cv2.VideoCapture, is_video: bool, conf: float, imgsz: int,
                     filter_name: Optional[str] = None,
                     banner: Optional[str] = None,
                     detect: bool = True):
    """Yield annotated JPEG frames as a multipart MJPEG stream.
    If detect=False, no YOLOE inference is run — raw feed only.
    If filter_name is given, keep only detections whose class name matches it.
    If banner is given, overlay a fixed text banner (e.g. 'SEARCHING: PEN')."""
    _STREAM_STOP.clear()
    t_prev = time.time()
    fps = 0.0
    fname_l = filter_name.lower() if filter_name else None
    try:
        while not _STREAM_STOP.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    break                          # end of file
                time.sleep(0.05)
                continue

            if detect:
                try:
                    dets = vision.detect_product(frame, conf=conf)
                except Exception as e:
                    dets = []
                    cv2.putText(frame, f"detect error: {e}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if fname_l is not None:
                    dets = [d for d in dets if d.get("name", "").lower() == fname_l]
                ann = vision.annotated(frame, dets)
            else:
                dets = []
                ann = frame

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t_prev = now
            cv2.rectangle(ann, (0, 0), (230, 55), (0, 0, 0), -1)
            cv2.putText(ann, f"FPS: {fps:.1f}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if detect:
                cv2.putText(ann, f"targets: {len(dets)}", (10, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                metrics.observe_detections(dets, fps)
            else:
                cv2.putText(ann, "raw feed (no detect)", (10, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            if banner:
                h, w = ann.shape[:2]
                b_color = (0, 200, 0) if dets else (0, 100, 255)
                (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(ann, (w - tw - 20, 0), (w, th + 20), b_color, -1)
                cv2.putText(ann, banner, (w - tw - 10, th + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            ok, jpg = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            b = jpg.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(b)).encode() + b"\r\n\r\n"
                   + b + b"\r\n")
    finally:
        cap.release()


def _apply_focus(filter_name: Optional[str]):
    """If a filter is requested, put the model into single-class mode for that
    product (accurate). Otherwise restore multi-product mode."""
    if filter_name:
        ok = vision.focus_on(filter_name)
        if not ok:
            print(f"[warn] focus_on({filter_name}) — no such locked product")
    else:
        vision.restore_all()


@app.get("/api/stream/webcam")
def stream_webcam(idx: int = 0, conf: float = 0.05, imgsz: int = 640,
                  filter: Optional[str] = None):
    cap = cv2.VideoCapture(int(idx), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise HTTPException(400, f"cannot open webcam {idx}")
    detect = filter is not None            # only run YOLOE on Section 3 (pick) previews
    focused = False
    if filter:
        focused = vision.focus_on(filter)
    banner = None
    if filter:
        banner = f"SEARCHING: {filter.upper()}" if focused else f"NOT LOCKED: {filter.upper()}"
    return StreamingResponse(
        _mjpeg_generator(cap, is_video=False, conf=conf, imgsz=imgsz,
                         banner=banner, detect=detect),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/video")
def stream_video(path: str, conf: float = 0.15, imgsz: int = 640,
                 filter: Optional[str] = None):
    if not Path(path).exists():
        raise HTTPException(404, f"not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, f"cannot open video: {path}")
    detect = filter is not None
    if filter:
        vision.focus_on(filter)
    return StreamingResponse(
        _mjpeg_generator(cap, is_video=True, conf=conf, imgsz=imgsz, detect=detect),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─── websocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _CLIENTS.add(ws)
    try:
        if _ROBOT is not None:
            await ws.send_text(json.dumps({"type": "pose", "state": _ROBOT.state.model_dump()}))
        while True:
            await ws.receive_text()      # heartbeat / discard
    except WebSocketDisconnect:
        pass
    finally:
        _CLIENTS.discard(ws)


# ─── uvicorn entry ───────────────────────────────────────────────────────────

def run():
    import uvicorn
    uvicorn.run("warehouse.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
