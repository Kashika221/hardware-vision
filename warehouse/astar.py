"""A* on a 2D occupancy grid (8-connected)."""
from __future__ import annotations

import heapq
import numpy as np

from .grid import FREE, OBSTACLE


NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
]


def _h(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (1.4142 - 2) * min(dr, dc)     # octile


def astar(static: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    if static[goal] == OBSTACLE:
        raise ValueError("goal is an obstacle")
    if start == goal:
        return [start]

    rows, cols = static.shape
    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return list(reversed(path))

        for dr, dc, step_cost in NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if static[nr, nc] == OBSTACLE:
                continue
            # prevent diagonal cutting through corners
            if dr != 0 and dc != 0:
                if static[cur[0] + dr, cur[1]] == OBSTACLE or static[cur[0], cur[1] + dc] == OBSTACLE:
                    continue
            ng = g_score[cur] + step_cost
            nb = (nr, nc)
            if ng < g_score.get(nb, float("inf")):
                came_from[nb] = cur
                g_score[nb] = ng
                f = ng + _h(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    raise RuntimeError(f"no path from {start} to {goal}")
