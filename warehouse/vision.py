"""Vision — QR shelf verification (pyzbar) + multi-product YOLOE detection.

Every stocked product contributes ONE visual prompt (reference image + bbox).
All prompts are stacked into a single set_classes() call on YOLOE, so the live
feed detects every product at once, each with its own name and colour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

try:
    from pyzbar.pyzbar import decode as _zbar_decode
except Exception:
    _zbar_decode = None

from ultralytics import YOLOE


# ─── QR / barcode ────────────────────────────────────────────────────────────

def scan_qr(image_bgr: np.ndarray) -> list[str]:
    if _zbar_decode is None:
        return _scan_qr_opencv(image_bgr)
    codes = _zbar_decode(image_bgr)
    return [c.data.decode("utf-8", errors="ignore") for c in codes]


def _scan_qr_opencv(image_bgr: np.ndarray) -> list[str]:
    det = cv2.QRCodeDetector()
    data, _pts, _ = det.detectAndDecodeMulti(image_bgr)
    if not data:
        return []
    return [d for d in data if d]


def verify_shelf(image_bgr: np.ndarray, expected_qr: str) -> bool:
    return expected_qr in scan_qr(image_bgr)


# ─── YOLOE multi-product store ──────────────────────────────────────────────

_MODEL: YOLOE | None = None
_PROMPTS: list[dict] = []          # [{name, image_path, bbox}]
_NAMES: list[str] = []             # class-index → name


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(weights: str = "yoloe-v8s-seg.pt") -> YOLOE:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLOE(weights)
        _MODEL.to(_device())
    return _MODEL


def clear_prompts() -> None:
    global _PROMPTS, _NAMES
    _PROMPTS = []
    _NAMES = []


def _compute_vpe_one(model: YOLOE, image: np.ndarray, bbox: list[int]) -> torch.Tensor:
    """Return a visual-prompt embedding for one reference crop.

    Uses the VP predictor's `get_vpe(source)` method directly — this is the
    Ultralytics-supported path in 8.4.x."""
    from ultralytics.models.yolo.yoloe import (
        YOLOEVPDetectPredictor,
        YOLOEVPSegPredictor,
    )

    prompts = dict(
        bboxes=np.array([bbox], dtype=np.float32),
        cls=np.array([0], dtype=np.int64),
    )

    is_seg = "seg" in str(getattr(model, "ckpt_path", "")).lower()
    Predictor = YOLOEVPSegPredictor if is_seg else YOLOEVPDetectPredictor

    predictor = Predictor(overrides={
        "task": model.model.task,
        "mode": "predict",
        "save": False,
        "verbose": False,
        "batch": 1,
        "imgsz": 640,
        "conf": 0.25,
    })
    predictor.setup_model(model=model.model)
    predictor.set_prompts(prompts.copy())
    vpe = predictor.get_vpe(image)

    # Don't leave any predictor lingering on the main YOLOE instance
    model.predictor = None
    return vpe


def _augment_ref(img: np.ndarray, bbox: list[int]) -> list[tuple[np.ndarray, list[int]]]:
    """Return the original + a horizontal-flip augmentation so the visual prompt
    is more robust to pose changes. (More rotations tend to hurt with elongated
    objects like pens; flip covers the common case.)"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    aug = [(img, bbox)]
    fl = cv2.flip(img, 1)
    aug.append((fl, [w - x2, y1, w - x1, y2]))
    return aug


def _ensure_vpe(prompt: dict) -> torch.Tensor | None:
    """Compute (and cache) the VPE. Raises on failure with details so the
    caller can surface a meaningful error."""
    if prompt.get("vpe") is not None:
        return prompt["vpe"]
    if not Path(prompt["image_path"]).exists():
        raise FileNotFoundError(f"ref image missing: {prompt['image_path']}")
    img = cv2.imread(prompt["image_path"])
    if img is None:
        raise RuntimeError(f"cv2.imread returned None for {prompt['image_path']}")
    model = get_model()
    vpes: list[torch.Tensor] = []
    errors: list[str] = []
    for aug_img, aug_bbox in _augment_ref(img, prompt["bbox"]):
        try:
            v = _compute_vpe_one(model, aug_img, aug_bbox)
            if v is None:
                errors.append("compute returned None")
                continue
            vpes.append(v)
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append(f"{type(e).__name__}: {e}")
    if not vpes:
        raise RuntimeError(" | ".join(errors) or "all augmentations produced None")
    # Some VPE tensors may have subtle shape differences; align on (1,1,D)
    aligned = []
    for v in vpes:
        if v.dim() == 3:
            aligned.append(v)
        elif v.dim() == 2:
            aligned.append(v.unsqueeze(0))
        else:
            raise RuntimeError(f"unexpected VPE shape {tuple(v.shape)}")
    stacked = torch.stack(aligned, dim=0).mean(dim=0)
    prompt["vpe"] = stacked
    return stacked


_LAST_VPE_ERROR: str | None = None


def last_vpe_error() -> str | None:
    return _LAST_VPE_ERROR


def _rebuild_classes() -> None:
    """Register every stored prompt as its own class — uses cached VPEs."""
    global _LAST_VPE_ERROR
    _LAST_VPE_ERROR = None
    if not _PROMPTS:
        _NAMES.clear()
        return
    model = get_model()
    vpes, names, errs = [], [], []
    for p in _PROMPTS:
        try:
            vpe = _ensure_vpe(p)
        except Exception as e:
            import traceback
            traceback.print_exc()
            errs.append(f"{p['name']}: {type(e).__name__}: {e}")
            continue
        if vpe is None:
            errs.append(f"{p['name']}: VPE returned None")
            continue
        vpes.append(vpe)
        names.append(p["name"])
    if not vpes:
        _LAST_VPE_ERROR = "; ".join(errs) or "no augmentations produced a VPE"
        print(f"[vision] rebuild failed: {_LAST_VPE_ERROR}")
        return
    stacked = torch.cat(vpes, dim=1)      # (1, N, D)
    model.set_classes(names, stacked)
    model.predictor = None                # force fresh Detect/Seg predictor next call
    _NAMES.clear()
    _NAMES.extend(names)
    print(f"[vision] classes registered: {names}")


def add_product_prompt(name: str, image_path: str, bbox_xyxy: list[int] | None) -> str:
    """Register (or update) the visual prompt for `name`."""
    bbox = bbox_xyxy
    if bbox is None:
        im = cv2.imread(image_path)
        if im is None:
            raise FileNotFoundError(image_path)
        h, w = im.shape[:2]
        bbox = [0, 0, w, h]

    # Update existing or append
    found = False
    for p in _PROMPTS:
        if p["name"].lower() == name.lower():
            p["image_path"] = image_path
            p["bbox"] = bbox
            p["vpe"] = None                 # invalidate cache
            found = True
            break
    if not found:
        _PROMPTS.append({"name": name, "image_path": image_path, "bbox": bbox, "vpe": None})

    _rebuild_classes()
    return name


def rehydrate_from_products(products: list) -> None:
    """Rebuild _PROMPTS from persisted Product records (called on startup)."""
    clear_prompts()
    for p in products:
        if p.ref_image_path and Path(p.ref_image_path).exists():
            _PROMPTS.append({
                "name": p.name,
                "image_path": p.ref_image_path,
                "bbox": p.ref_bbox or None,
            })
    # Fill missing bboxes with full-image fallback
    for p in _PROMPTS:
        if p["bbox"] is None:
            im = cv2.imread(p["image_path"])
            if im is not None:
                h, w = im.shape[:2]
                p["bbox"] = [0, 0, w, h]
    _rebuild_classes()


def locked_names() -> list[str]:
    return list(_NAMES)


# ─── Focus / restore — for pick preview ─────────────────────────────────────

def focus_on(name: str) -> bool:
    """Set the model to detect ONLY this product — much more accurate than
    multi-class competition. Returns True if the target's VPE is now active."""
    prompt = next((p for p in _PROMPTS if p["name"].lower() == name.lower()), None)
    if prompt is None:
        print(f"[vision] focus_on: no prompt for '{name}' (known: {[p['name'] for p in _PROMPTS]})")
        return False
    vpe = _ensure_vpe(prompt)
    if vpe is None:
        return False
    model = get_model()
    model.set_classes([prompt["name"]], vpe)
    model.predictor = None
    _NAMES.clear()
    _NAMES.append(prompt["name"])
    print(f"[vision] focused on: {prompt['name']}")
    return True


def restore_all() -> None:
    """Re-register every stored prompt so the model detects all products again."""
    _rebuild_classes()


# ─── Detection + drawing ────────────────────────────────────────────────────

_COLOURS = [
    (0, 255,   0), (255, 120,   0), ( 30, 200, 255), (255,  60, 200),
    (255, 255,   0), (  0, 200, 255), (200, 100, 255), (100, 255,  60),
]


def _colour_for(idx: int) -> tuple[int, int, int]:
    return _COLOURS[idx % len(_COLOURS)]


def detect(image_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> list[dict]:
    """Return [{bbox, confidence, cls, name}] for every match."""
    if not _NAMES:
        return []
    model = get_model()
    results = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device=_device(),
    )[0]
    out = []
    if results.boxes is None or len(results.boxes) == 0:
        return out
    xyxy = results.boxes.xyxy.cpu().numpy().astype(int).tolist()
    confs = results.boxes.conf.cpu().numpy().tolist()
    clses = results.boxes.cls.cpu().numpy().astype(int).tolist()
    for b, c, k in zip(xyxy, confs, clses):
        name = _NAMES[k] if 0 <= k < len(_NAMES) else "target"
        out.append({"bbox": b, "confidence": float(c), "cls": int(k), "name": name})
    return out


# backwards-compat alias so older callers keep working
detect_product = detect


def annotated(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        colour = _colour_for(d.get("cls", 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)
        text = f"{d.get('name', 'target').upper()} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


# ─── QR sticker generator ───────────────────────────────────────────────────

def make_qr_png(payload: str) -> bytes:
    import io
    import qrcode
    img = qrcode.make(payload)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
