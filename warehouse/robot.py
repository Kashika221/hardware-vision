"""Simulated robot — steps along an A* path, broadcasts pose."""
from __future__ import annotations

import asyncio
import math

from . import metrics
from .astar import astar
from .grid import approach_cell, build_static_grid, cell_to_world, world_to_cell
from .models import RobotState, Warehouse


STEP_INTERVAL_S = 0.15   # per cell — 6-7 cells/sec, snappy for demo


class Robot:
    def __init__(self, wh: Warehouse):
        self.wh = wh
        self.state = RobotState(
            x_m=wh.robot_start_xy_m[0],
            y_m=wh.robot_start_xy_m[1],
        )
        self._on_update: list = []

    def on_update(self, cb):
        self._on_update.append(cb)

    async def _push(self):
        for cb in self._on_update:
            res = cb(self.state)
            if asyncio.iscoroutine(res):
                await res

    def current_cell(self) -> tuple[int, int]:
        return world_to_cell(self.wh, self.state.x_m, self.state.y_m)

    async def _walk(self, path: list[tuple[int, int]]):
        for row, col in path:
            x, y = cell_to_world(self.wh, row, col)
            # heading based on delta
            dx, dy = x - self.state.x_m, y - self.state.y_m
            if dx != 0 or dy != 0:
                self.state.theta_deg = math.degrees(math.atan2(dy, dx))
            self.state.x_m, self.state.y_m = x, y
            self.state.path_cells = path
            metrics.robot_step(x, y, row, col)
            await self._push()
            await asyncio.sleep(STEP_INTERVAL_S)

    async def goto_shelf(self, shelf_id: str, message: str = ""):
        from . import database

        shelf = database.get_shelf(shelf_id)
        if shelf is None:
            raise ValueError(f"unknown shelf {shelf_id}")

        static = build_static_grid(self.wh)
        goal = approach_cell(self.wh, shelf)
        start = self.current_cell()

        self.state.status = "moving"
        self.state.goal_shelf_id = shelf_id
        self.state.message = message or f"navigating to {shelf_id}"
        await self._push()

        path = astar(static, start, goal)
        await self._walk(path)

        self.state.status = "at_shelf"
        self.state.message = f"arrived at {shelf_id}"
        await self._push()

    async def return_home(self):
        static = build_static_grid(self.wh)
        start = self.current_cell()
        goal = world_to_cell(self.wh, *self.wh.robot_start_xy_m)
        self.state.status = "returning"
        self.state.message = "returning to home"
        await self._push()
        path = astar(static, start, goal)
        await self._walk(path)
        self.state.status = "idle"
        self.state.goal_shelf_id = None
        self.state.goal_product = None
        self.state.path_cells = []
        self.state.message = "idle"
        await self._push()
