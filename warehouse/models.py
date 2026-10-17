"""Domain models — Pydantic v2."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Shelf(BaseModel):
    id: str                          # "S01"
    qr: str                          # "QR-S01"
    x_m: float                       # world coord, meters (center)
    y_m: float
    w_m: float                       # footprint on floor
    h_m: float


class Product(BaseModel):
    name: str
    barcode: str
    shelf_id: str
    qty: int = 0
    ref_image_path: Optional[str] = None   # YOLOE visual prompt reference
    ref_bbox: Optional[list[int]] = None   # [x1,y1,x2,y2] in ref image px


class Warehouse(BaseModel):
    width_m: float
    height_m: float
    cell_size_m: float = 0.10        # 10 cm/cell
    shelves: list[Shelf] = []
    robot_start_xy_m: tuple[float, float] = (0.5, 0.5)

    @property
    def rows(self) -> int:
        return int(round(self.height_m / self.cell_size_m))

    @property
    def cols(self) -> int:
        return int(round(self.width_m / self.cell_size_m))


class InventoryDB(BaseModel):
    warehouse: Optional[Warehouse] = None
    products: list[Product] = []
    next_shelf_index: int = 0        # round-robin cursor


class RobotState(BaseModel):
    x_m: float = 0.5
    y_m: float = 0.5
    theta_deg: float = 0.0
    status: Literal["idle", "moving", "at_shelf", "picking", "placing", "returning", "error"] = "idle"
    goal_shelf_id: Optional[str] = None
    goal_product: Optional[str] = None
    path_cells: list[tuple[int, int]] = []
    message: str = ""


class Command(BaseModel):
    kind: Literal["stock", "pick"]
    product_name: str
    barcode: Optional[str] = None
