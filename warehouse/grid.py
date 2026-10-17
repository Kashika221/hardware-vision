"""Occupancy grid — 0 free, 1 obstacle, 2 robot (dynamic)."""
from __future__ import annotations

import numpy as np

from .models import Warehouse


FREE, OBSTACLE, ROBOT = 0, 1, 2


def build_static_grid(wh: Warehouse) -> np.ndarray:
    """0=free, 1=obstacle. Robot position is applied dynamically on top."""
    g = np.zeros((wh.rows, wh.cols), dtype=np.uint8)
    for s in wh.shelves:
        r0, c0 = world_to_cell(wh, s.x_m - s.w_m / 2, s.y_m - s.h_m / 2)
        r1, c1 = world_to_cell(wh, s.x_m + s.w_m / 2, s.y_m + s.h_m / 2)
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(wh.rows - 1, r1); c1 = min(wh.cols - 1, c1)
        g[r0:r1 + 1, c0:c1 + 1] = OBSTACLE
    return g


def world_to_cell(wh: Warehouse, x_m: float, y_m: float) -> tuple[int, int]:
    """World (x=col-direction, y=row-direction, origin bottom-left) → (row, col).
    Row 0 is TOP of the grid array (matches image convention)."""
    col = int(round(x_m / wh.cell_size_m))
    # invert y so bigger y = lower row index (like a math plot)
    row = int(round((wh.height_m - y_m) / wh.cell_size_m))
    return row, col


def cell_to_world(wh: Warehouse, row: int, col: int) -> tuple[float, float]:
    x = col * wh.cell_size_m
    y = wh.height_m - row * wh.cell_size_m
    return x, y


def approach_cell(wh: Warehouse, shelf) -> tuple[int, int]:
    """Pick a free cell adjacent to the shelf as the approach/goal cell."""
    static = build_static_grid(wh)
    sr, sc = world_to_cell(wh, shelf.x_m, shelf.y_m)
    # scan a small ring around the shelf for the first FREE cell
    for radius in range(1, 20):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                r, c = sr + dr, sc + dc
                if 0 <= r < wh.rows and 0 <= c < wh.cols and static[r, c] == FREE:
                    return r, c
    raise RuntimeError(f"no reachable approach cell for {shelf.id}")


def render_dynamic(static: np.ndarray, robot_rc: tuple[int, int]) -> np.ndarray:
    g = static.copy()
    r, c = robot_rc
    if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and g[r, c] != OBSTACLE:
        g[r, c] = ROBOT
    return g
