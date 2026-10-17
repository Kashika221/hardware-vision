"""Runtime metrics store. Every part of the system pushes small events into
here; the dashboard reads a JSON snapshot from `/api/metrics`."""
from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from datetime import datetime
from typing import Optional


_LOCK = threading.RLock()
_START_TS = time.time()

# ─── counters ────────────────────────────────────────────────────────────────
_distance_m: float = 0.0
_cell_visits: Counter[tuple[int, int]] = Counter()

_tasks_stock: int = 0
_tasks_pick: int = 0
_last_task_ms: list[float] = []            # rolling window of task durations
_task_start_ts: Optional[float] = None

# per-product detection totals (only when vision.detect returns matches)
_detections: Counter[str] = Counter()
_confidence_sum: dict[str, float] = {}

# rolling samples
_fps_samples: deque[tuple[float, float]] = deque(maxlen=120)   # (ts, fps)
_confidence_samples: deque[tuple[float, float]] = deque(maxlen=120)

# activity log — human-readable timeline
_activity: deque[dict] = deque(maxlen=100)


# ─── helpers ────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_event(kind: str, message: str, **extra) -> None:
    with _LOCK:
        _activity.appendleft({
            "ts": _now_iso(),
            "kind": kind,
            "message": message,
            **extra,
        })


# ─── robot ──────────────────────────────────────────────────────────────────

_last_xy: Optional[tuple[float, float]] = None


def robot_step(x_m: float, y_m: float, row: int, col: int) -> None:
    """Called on every simulated robot pose update."""
    global _last_xy, _distance_m
    with _LOCK:
        if _last_xy is not None:
            dx = x_m - _last_xy[0]
            dy = y_m - _last_xy[1]
            _distance_m += math.hypot(dx, dy)
        _last_xy = (x_m, y_m)
        _cell_visits[(row, col)] += 1


def reset_last_pose() -> None:
    global _last_xy
    with _LOCK:
        _last_xy = None


# ─── tasks ──────────────────────────────────────────────────────────────────

def task_start(kind: str, product: str, shelf: str | None = None) -> None:
    global _task_start_ts
    with _LOCK:
        _task_start_ts = time.time()
        log_event(kind, f"{kind.upper()} {product}" + (f" → {shelf}" if shelf else ""),
                  product=product, shelf=shelf)


def task_end(kind: str) -> None:
    global _task_start_ts, _tasks_stock, _tasks_pick
    with _LOCK:
        if _task_start_ts is not None:
            dur = (time.time() - _task_start_ts) * 1000.0
            _last_task_ms.append(dur)
            if len(_last_task_ms) > 60:
                del _last_task_ms[0]
        _task_start_ts = None
        if kind == "stock":
            _tasks_stock += 1
        elif kind == "pick":
            _tasks_pick += 1


# ─── vision ─────────────────────────────────────────────────────────────────

def observe_detections(dets: list[dict], fps: float) -> None:
    with _LOCK:
        now = time.time()
        _fps_samples.append((now, fps))
        if dets:
            avg_c = sum(d["confidence"] for d in dets) / len(dets)
            _confidence_samples.append((now, avg_c))
        for d in dets:
            name = d.get("name", "target")
            _detections[name] += 1
            _confidence_sum[name] = _confidence_sum.get(name, 0.0) + d["confidence"]


# ─── snapshot ───────────────────────────────────────────────────────────────

def snapshot(warehouse=None) -> dict:
    with _LOCK:
        n_tasks = _tasks_stock + _tasks_pick
        avg_task_ms = (sum(_last_task_ms) / len(_last_task_ms)) if _last_task_ms else 0
        avg_conf_by_product = {
            k: (v / _detections[k]) if _detections[k] else 0.0
            for k, v in _confidence_sum.items()
        }
        # normalise timestamps to seconds-since-start for compact plotting
        t0 = _START_TS
        fps_series = [(round(t - t0, 1), round(v, 1)) for t, v in _fps_samples]
        conf_series = [(round(t - t0, 1), round(v, 3)) for t, v in _confidence_samples]

        heatmap = None
        if warehouse is not None:
            heat = [[0] * warehouse.cols for _ in range(warehouse.rows)]
            for (r, c), v in _cell_visits.items():
                if 0 <= r < warehouse.rows and 0 <= c < warehouse.cols:
                    heat[r][c] = v
            heatmap = heat

        return {
            "uptime_s": round(time.time() - _START_TS, 1),
            "distance_m": round(_distance_m, 2),
            "tasks_total": n_tasks,
            "tasks_stock": _tasks_stock,
            "tasks_pick": _tasks_pick,
            "avg_task_ms": round(avg_task_ms, 0),
            "detections_total": sum(_detections.values()),
            "detections_by_product": dict(_detections),
            "avg_confidence_by_product": {k: round(v, 3) for k, v in avg_conf_by_product.items()},
            "fps_series": fps_series,
            "confidence_series": conf_series,
            "activity": list(_activity)[:30],
            "cell_visit_heatmap": heatmap,
        }


def reset_all() -> None:
    """Wipe every metric — used when a new warehouse is imported."""
    global _distance_m, _cell_visits, _tasks_stock, _tasks_pick, _last_task_ms
    global _task_start_ts, _detections, _confidence_sum, _last_xy
    with _LOCK:
        _distance_m = 0.0
        _cell_visits = Counter()
        _tasks_stock = 0
        _tasks_pick = 0
        _last_task_ms = []
        _task_start_ts = None
        _detections = Counter()
        _confidence_sum = {}
        _fps_samples.clear()
        _confidence_samples.clear()
        _activity.clear()
        _last_xy = None
