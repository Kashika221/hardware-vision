"""
Warehouse AMR — Target-Lock Vision
Single-model, zero-training, image-prompted object detection.

Flow:
    Phase 1  Lock the target  (from an image OR a video frame)
    Phase 2  Test detection   (on webcam OR a video OR a still image)

Run:
    python target_lock.py
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLOE


# ─────────────────────────── interactive CLI ────────────────────────────────

def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or (default or "")


def choose(prompt: str, options: list[str]) -> int:
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        raw = input("Choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"Enter a number 1–{len(options)}.")


def list_files(exts, subdirs=(".", "videos", "refs", "images")):
    hits = []
    for sd in subdirs:
        for e in exts:
            hits.extend(glob.glob(os.path.join(sd, e)))
    return sorted({os.path.normpath(h) for h in hits})


def pick_file(kind: str) -> str:
    if kind == "video":
        exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    files = list_files(exts)
    if not files:
        return ask(f"{kind.capitalize()} path")

    idx = choose(f"Select a {kind} (or last option to type a path):",
                 files + [f"<enter {kind} path manually>"])
    if idx == len(files):
        return ask(f"{kind.capitalize()} path")
    return files[idx]


# ─────────────────────────── model + inference ──────────────────────────────

def load_model(weights: str, device: str) -> YOLOE:
    print(f"\n[boot] loading {weights} on {device} ...")
    model = YOLOE(weights)
    model.to(device)
    print("[boot] ready")
    return model


def pick_bbox(frame: np.ndarray, prompt: str = "target") -> list[int] | None:
    win = f"Draw box around {prompt} — ENTER to confirm, C to cancel"
    roi = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return [int(x), int(y), int(x + w), int(y + h)]


def set_visual_prompt(model: YOLOE, ref_img: np.ndarray, bbox_xyxy: list[int]) -> None:
    from ultralytics.models.yolo.yoloe import (
        YOLOEVPDetectPredictor,
        YOLOEVPSegPredictor,
    )
    is_seg = "seg" in str(getattr(model, "ckpt_path", "")).lower()
    vp_predictor = YOLOEVPSegPredictor if is_seg else YOLOEVPDetectPredictor

    visual_prompts = dict(
        bboxes=np.array([bbox_xyxy], dtype=np.float32),
        cls=np.array([0], dtype=np.int64),
    )
    model.predict(
        source=ref_img,
        refer_image=ref_img,
        visual_prompts=visual_prompts,
        predictor=vp_predictor,
        imgsz=640,
        conf=0.25,
        verbose=False,
    )
    try:
        vpe = model.predictor.vpe
        model.set_classes(["target"], vpe)
    except AttributeError:
        pass
    print(f"[prompt] target locked: bbox={bbox_xyxy}")


def draw(frame, boxes, fps=None):
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"TARGET {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if fps is not None:
        cv2.rectangle(frame, (0, 0), (240, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "r=re-lock  q=quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def run_predict(model, frame, imgsz, conf, device):
    results = model.predict(source=frame, imgsz=imgsz, conf=conf,
                            verbose=False, device=device)[0]
    boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append((x1, y1, x2, y2, float(c)))
    return boxes


# ────────────────────────── PHASE 1: get target ─────────────────────────────

def get_frame_from_video(path: str) -> np.ndarray:
    """Let the user scrub to a frame in the video with a trackbar."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1

    win = "Scrub to a frame — SPACE to accept, Q to cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", win, 0, max(total - 1, 1), lambda _v: None)

    chosen = None
    while True:
        idx = cv2.getTrackbarPos("frame", win)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        preview = frame.copy()
        cv2.putText(preview, f"frame {idx}/{total-1}  SPACE=accept  Q=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, preview)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            chosen = frame
            break
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyWindow(win)
    if chosen is None:
        raise SystemExit("no frame chosen")
    return chosen


def get_frame_from_webcam() -> np.ndarray:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("cannot open webcam")
    win = "Hold up the target — SPACE to snap, Q to cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    chosen = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        preview = frame.copy()
        cv2.putText(preview, "SPACE=snap  Q=cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            chosen = frame
            break
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyWindow(win)
    if chosen is None:
        raise SystemExit("no snapshot taken")
    return chosen


def phase1_lock_target(model: YOLOE):
    print("\n" + "─" * 60)
    print("  PHASE 1 — Lock the target")
    print("─" * 60)

    idx = choose("How do you want to provide the target?", [
        "Reference image file",
        "Reference video file (scrub to a good frame)",
        "Webcam snapshot",
    ])

    if idx == 0:
        path = pick_file("image")
        ref = cv2.imread(path)
        if ref is None:
            raise SystemExit(f"cannot read: {path}")
    elif idx == 1:
        path = pick_file("video")
        ref = get_frame_from_video(path)
    else:
        ref = get_frame_from_webcam()

    bbox = pick_bbox(ref, "target")
    if bbox is None:
        raise SystemExit("no target selected")

    set_visual_prompt(model, ref, bbox)


# ────────────────────────── PHASE 2: test model ─────────────────────────────

def open_source(src: str) -> cv2.VideoCapture:
    if src.isdigit():
        return cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
    return cv2.VideoCapture(src)


def stream_loop(model, cap, imgsz, conf, device, allow_relock=True):
    win = "Warehouse AMR — Target Lock"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = run_predict(model, frame, imgsz, conf, device)

        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        fps = 0.9 * fps + 0.1 * inst if fps else inst
        t_prev = now

        cv2.imshow(win, draw(frame, boxes, fps))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if allow_relock and key == ord("r"):
            ok, ref = cap.read()
            if ok:
                new_bbox = pick_bbox(ref, "new target")
                if new_bbox is not None:
                    set_visual_prompt(model, ref, new_bbox)

    cap.release()
    cv2.destroyAllWindows()


def test_on_image(model, imgsz, conf, device):
    path = pick_file("image")
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read: {path}")

    t0 = time.time()
    boxes = run_predict(model, img, imgsz, conf, device)
    dt_ms = (time.time() - t0) * 1000

    out = draw(img.copy(), boxes)
    cv2.putText(out, f"{len(boxes)} target(s)   {dt_ms:.0f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    save = os.path.splitext(path)[0] + "_detected.jpg"
    cv2.imwrite(save, out)
    print(f"[saved] {save}   ({len(boxes)} detection(s), {dt_ms:.0f} ms)")

    win = "Detection — press any key to close"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def phase2_test(model: YOLOE, imgsz: int, conf: float, device: str):
    while True:
        print("\n" + "─" * 60)
        print("  PHASE 2 — Test detection")
        print("─" * 60)

        idx = choose("Where do you want to run detection?", [
            "Webcam (real-time)",
            "Video file",
            "Single image",
            "Re-lock a new target",
            "Quit",
        ])

        if idx == 0:
            cam = ask("Camera index", "0")
            cap = open_source(cam)
            if not cap.isOpened():
                print(f"[!] cannot open camera {cam}")
                continue
            stream_loop(model, cap, imgsz, conf, device, allow_relock=True)

        elif idx == 1:
            path = pick_file("video")
            cap = open_source(path)
            if not cap.isOpened():
                print(f"[!] cannot open {path}")
                continue
            stream_loop(model, cap, imgsz, conf, device, allow_relock=False)

        elif idx == 2:
            test_on_image(model, imgsz, conf, device)

        elif idx == 3:
            phase1_lock_target(model)

        else:
            print("[exit] bye")
            return


# ────────────────────────────────── main ─────────────────────────────────────

def main():
    print("=" * 60)
    print("  Warehouse AMR — Target-Lock Vision (YOLOE)")
    print("=" * 60)

    weights = ask("YOLOE weights", "yoloe-v8s-seg.pt")
    imgsz = int(ask("Inference size (px)", "640"))
    conf = float(ask("Confidence threshold", "0.25"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = load_model(weights, device)

    phase1_lock_target(model)
    phase2_test(model, imgsz, conf, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] bye")
        sys.exit(0)
