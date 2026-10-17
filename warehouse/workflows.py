"""High-level workflows: STOCK a new product, PICK an existing one."""
from __future__ import annotations

from . import database, metrics, vision
from .robot import Robot


async def stock(robot: Robot, product_name: str,
                barcode: str | None, ref_image_path: str | None,
                ref_bbox: list[int] | None = None,
                shelf_id: str | None = None) -> dict:
    """Assign a shelf (explicit shelf_id if given, else round-robin), navigate, place."""
    product = database.upsert_product(product_name, barcode, ref_image_path,
                                      ref_bbox, shelf_id=shelf_id)
    shelf = database.get_shelf(product.shelf_id)
    if shelf is None:
        return {"ok": False, "error": f"shelf {product.shelf_id} missing"}

    metrics.task_start("stock", product_name, shelf.id)
    await robot.goto_shelf(shelf.id, f"stocking {product_name} → {shelf.id}")

    # QR verification would happen from a real camera; sim assumes success.
    robot.state.status = "placing"
    robot.state.message = f"placing {product_name}"
    await robot._push()
    await _sleep(1.0)

    await robot.return_home()
    metrics.task_end("stock")
    return {"ok": True, "product": product.model_dump(), "shelf": shelf.model_dump()}


async def pick(robot: Robot, product_name: str) -> dict:
    product = database.find_product(product_name)
    if product is None:
        return {"ok": False, "error": f"unknown product '{product_name}'"}
    if product.qty <= 0:
        return {"ok": False, "error": f"'{product_name}' out of stock"}

    shelf = database.get_shelf(product.shelf_id)
    if shelf is None:
        return {"ok": False, "error": f"shelf {product.shelf_id} missing"}

    metrics.task_start("pick", product_name, shelf.id)
    await robot.goto_shelf(shelf.id, f"picking {product_name} from {shelf.id}")

    robot.state.status = "picking"
    robot.state.goal_product = product_name
    robot.state.message = f"picking {product_name}"
    await robot._push()
    await _sleep(1.0)

    database.decrement_product(product_name)
    await robot.return_home()
    # Pick preview left YOLOE in single-target focus mode — restore
    # all-products tracking now that the robot is safely home.
    try:
        vision.restore_all()
        print(f"[workflows] pick done — YOLOE re-tracking: {vision.locked_names()}")
    except Exception as e:
        print(f"[warn] restore_all after pick failed: {e}")
    metrics.task_end("pick")
    return {"ok": True, "product": product.model_dump(), "shelf": shelf.model_dump()}


async def _sleep(s: float):
    import asyncio
    await asyncio.sleep(s)
