"""JSON-backed persistence."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .models import InventoryDB, Product, Warehouse

_LOCK = threading.RLock()
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inventory.json"


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def load() -> InventoryDB:
    with _LOCK:
        _ensure_dir()
        if not DB_PATH.exists():
            return InventoryDB()
        return InventoryDB.model_validate_json(DB_PATH.read_text(encoding="utf-8"))


def save(db: InventoryDB) -> None:
    with _LOCK:
        _ensure_dir()
        DB_PATH.write_text(db.model_dump_json(indent=2), encoding="utf-8")


# ─── high-level helpers ───────────────────────────────────────────────────────

def set_warehouse(wh: Warehouse, reset_inventory: bool = True) -> InventoryDB:
    """Persist the warehouse. If reset_inventory (default), wipe all products so
    every new map starts with a fresh inventory."""
    db = load()
    db.warehouse = wh
    db.next_shelf_index = 0
    if reset_inventory:
        db.products = []
    save(db)
    return db


def find_product(name: str) -> Product | None:
    db = load()
    name_l = name.lower().strip()
    for p in db.products:
        if p.name.lower() == name_l:
            return p
    return None


def round_robin_shelf(db: InventoryDB) -> str:
    if not db.warehouse or not db.warehouse.shelves:
        raise RuntimeError("no warehouse / shelves defined")
    shelf = db.warehouse.shelves[db.next_shelf_index % len(db.warehouse.shelves)]
    db.next_shelf_index = (db.next_shelf_index + 1) % len(db.warehouse.shelves)
    return shelf.id


def upsert_product(name: str, barcode: str | None,
                   ref_image_path: str | None,
                   ref_bbox: list[int] | None = None,
                   shelf_id: str | None = None) -> Product:
    db = load()
    existing = None
    for p in db.products:
        if p.name.lower() == name.lower():
            existing = p
            break

    if existing:
        existing.qty += 1
        if ref_image_path:
            existing.ref_image_path = ref_image_path
        if ref_bbox:
            existing.ref_bbox = ref_bbox
        if shelf_id and shelf_id != existing.shelf_id:
            # user chose to relocate the product to a different shelf
            if get_shelf(shelf_id) is not None:
                existing.shelf_id = shelf_id
        save(db)
        return existing

    # New product: honour explicit shelf_id if valid, else round-robin
    chosen = None
    if shelf_id and get_shelf(shelf_id) is not None:
        chosen = shelf_id
    else:
        chosen = round_robin_shelf(db)

    product = Product(
        name=name,
        barcode=barcode or f"P-{abs(hash(name)) % 10**9:09d}",
        shelf_id=chosen,
        qty=1,
        ref_image_path=ref_image_path,
        ref_bbox=ref_bbox,
    )
    db.products.append(product)
    save(db)
    return product


def decrement_product(name: str) -> Product | None:
    db = load()
    for p in db.products:
        if p.name.lower() == name.lower():
            if p.qty > 0:
                p.qty -= 1
                save(db)
            return p
    return None


def get_shelf(shelf_id: str) -> "Shelf | None":  # noqa: F821
    db = load()
    if not db.warehouse:
        return None
    for s in db.warehouse.shelves:
        if s.id == shelf_id:
            return s
    return None
