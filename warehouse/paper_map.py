"""Paper drawing → digital warehouse.

Standardized template (pen on plain paper):
    - The OUTER rectangle is the warehouse boundary.
    - Each INNER rectangle is a shelf.
    - Draw a filled dot / small solid circle where the robot should start.

The camera can be angled; the outer rectangle is used for perspective correction
so the imported map is always axis-aligned.

Shelves are auto-labeled S01, S02, ... top-to-bottom then left-to-right.
The user can rename them in the dashboard afterwards.
"""
from __future__ import annotations

import io
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import Shelf, Warehouse


# ─────────────────── SYNTHESIZE a pen-paper style test map ──────────────────

CREAM = (245, 238, 220)
INK = (35, 30, 25)


def synthesize_paper_map(
    width_m: float = 10.0,
    height_m: float = 6.0,
    n_shelves: int = 4,
    seed: int | None = None,
    out_size_px: tuple[int, int] = (1200, 800),
) -> Image.Image:
    """Draw a fake pen-and-paper warehouse sketch."""
    rng = random.Random(seed)
    W, H = out_size_px
    img = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(img)

    # Slightly rough boundary
    m = 60
    box = [(m, m), (W - m, m), (W - m, H - m), (m, H - m)]
    for i in range(4):
        p, q = box[i], box[(i + 1) % 4]
        draw.line([p, q], fill=INK, width=4)

    # Place shelves in a rough grid, not overlapping
    inner_w = (W - 2 * m) - 80
    inner_h = (H - 2 * m) - 80
    cols = int(math.ceil(math.sqrt(n_shelves)))
    rows = int(math.ceil(n_shelves / cols))
    cell_w = inner_w / cols
    cell_h = inner_h / rows

    shelves_px = []
    for i in range(n_shelves):
        r, c = i // cols, i % cols
        cx = m + 40 + c * cell_w + cell_w / 2
        cy = m + 40 + r * cell_h + cell_h / 2
        sw = cell_w * rng.uniform(0.45, 0.65)
        sh = cell_h * rng.uniform(0.35, 0.55)
        x0, y0, x1, y1 = cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2
        draw.rectangle([x0, y0, x1, y1], outline=INK, width=3)
        shelves_px.append((x0, y0, x1, y1))
        # small ID hint (optional, cosmetic — importer doesn't OCR it)
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        draw.text((x0 + 8, y0 + 6), f"S{i+1:02d}", fill=INK, font=font)

    # Robot start = filled dot near bottom-left interior
    rx, ry = m + 60, H - m - 60
    draw.ellipse([rx - 12, ry - 12, rx + 12, ry + 12], fill=INK)
    try:
        font_r = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font_r = ImageFont.load_default()
    draw.text((rx + 18, ry - 12), "R", fill=INK, font=font_r)

    # Dimension labels
    draw.text((W // 2 - 30, 20), f"{width_m} m", fill=INK, font=font_r)
    draw.text((20, H // 2), f"{height_m} m", fill=INK, font=font_r)
    return img


# ─────────────────── IMPORT a real photo of a paper map ─────────────────────

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return [tl, tr, br, bl] ordered corners."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_outer_quad(gray: np.ndarray) -> np.ndarray | None:
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15 * gray.size:
            return _order_quad(approx)
    return None


def _warp_to_axis(bgr: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h))


def import_paper_map(
    image_bytes: bytes,
    width_m: float,
    height_m: float,
    cell_size_m: float = 0.10,
) -> tuple[Warehouse, np.ndarray]:
    """Return (Warehouse, warped_bgr_preview)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot decode image")

    # Resize preserving aspect for stable processing
    max_side = 1400
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = _find_outer_quad(gray)

    # Warp so 1 m == fixed px (proportional to warehouse aspect)
    out_w = 1200
    out_h = int(out_w * (height_m / width_m))

    if quad is not None:
        warped = _warp_to_axis(bgr, quad, out_w, out_h)
    else:
        # Fall back: assume the drawing already fills the frame
        warped = cv2.resize(bgr, (out_w, out_h))

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Binary — ink dark on light paper
    _, bw = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Find rectangles inside the warped image (shelves)
    contours, hier = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shelves_px = []
    dot_center = None
    dot_r = 0
    total_area = warped.shape[0] * warped.shape[1]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 800 or area > 0.6 * total_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        circularity = area / (math.pi * radius * radius + 1e-6)

        if len(approx) == 4:
            x, y, w2, h2 = cv2.boundingRect(approx)
            # skip the outer boundary itself
            if w2 > 0.9 * warped.shape[1] and h2 > 0.9 * warped.shape[0]:
                continue
            shelves_px.append((x, y, x + w2, y + h2))
        elif circularity > 0.7 and radius < 40 and radius > 5:
            # candidate for robot start dot — pick the most circular smallest
            if dot_center is None or radius < dot_r:
                dot_center = (cx, cy)
                dot_r = radius

    # de-dup overlapping shelves (nested contours)
    shelves_px = _dedupe_boxes(shelves_px)
    shelves_px.sort(key=lambda b: (b[1], b[0]))   # top-to-bottom, then left-to-right

    # px → world (bottom-left origin, y up)
    W_px, H_px = warped.shape[1], warped.shape[0]

    def px_to_world(x_px, y_px):
        return (x_px / W_px) * width_m, (1 - y_px / H_px) * height_m

    shelves: list[Shelf] = []
    for i, (x0, y0, x1, y1) in enumerate(shelves_px, start=1):
        cx_px, cy_px = (x0 + x1) / 2, (y0 + y1) / 2
        wx, wy = px_to_world(cx_px, cy_px)
        w_m = (x1 - x0) / W_px * width_m
        h_m = (y1 - y0) / H_px * height_m
        sid = f"S{i:02d}"
        shelves.append(Shelf(id=sid, qr=f"QR-{sid}", x_m=wx, y_m=wy, w_m=w_m, h_m=h_m))

    if dot_center is not None:
        rx, ry = px_to_world(*dot_center)
    else:
        rx, ry = 0.5, 0.5

    wh = Warehouse(
        width_m=width_m,
        height_m=height_m,
        cell_size_m=cell_size_m,
        shelves=shelves,
        robot_start_xy_m=(rx, ry),
    )
    return wh, warped


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for b in boxes:
        if not any(_iou(b, k) > 0.5 for k in kept):
            kept.append(b)
    return kept


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-6)


def pil_to_bytes(im: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()
