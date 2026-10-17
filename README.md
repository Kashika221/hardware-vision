# Autonomous Inventory Management Robot — Large Warehouse

I am designing an **autonomous inventory management robot for large warehouses/inventories containing multiple shelves/racks and many different products**.

The robot should perform two main operations:

1. **Stocking / Adding a product**
2. **Retrieving / Picking a product**

The system should combine:

- Inventory database
- Product identification
- Paper-to-digital warehouse mapping
- Coordinate-based robot navigation
- Dynamic robot localization
- QR/barcode-based shelf identification
- Computer vision/object detection
- Robotic arm
- Backend/API

For the current version, **ignore shelf capacity, product dimensions, weight, and spatial optimization**. When a new product is added, use a simple **round-robin shelf allocation strategy**.

---

# 1. Core Concept

The system should maintain a digital representation of the warehouse as a **coordinate-based occupancy map**.

The map represents:

```text
0 = free/traversable space
1 = obstacle/wall/shelf
2 = current robot position
```

For example:

```text
0 0 0 0 0 0 0 0
0 1 1 1 0 0 0 0
0 1 1 1 0 1 0 0
0 0 0 0 0 1 0 0
0 0 0 2 0 0 0 0
```

Here:

- `1` represents an obstacle such as a shelf or wall.
- `0` represents space through which the robot can move.
- `2` represents the robot's current position.

The **underlying warehouse layout remains unchanged**. The value `2` is a dynamic representation of the robot's current position.

---

# 2. Dynamic Robot Position

The robot's position should be continuously updated while it moves.

For example, initially:

```text
0 0 0 0 0
0 0 0 0 0
0 0 2 0 0
0 0 0 0 0
0 0 0 0 0
```

The robot moves one cell:

```text
0 0 0 0 0
0 0 0 0 0
0 0 0 2 0
0 0 0 0 0
0 0 0 0 0
```

The previous robot position becomes its original underlying value, normally `0` because it was free space.

Therefore, the system continuously performs:

```text
Previous robot position:
2 → 0

New robot position:
0 → 2
```

If the robot moves through multiple cells:

```text
Time 1:

0 0 0
0 2 0
0 0 0


Time 2:

0 0 0
0 0 2
0 0 0


Time 3:

0 0 0
0 0 0
0 0 2
```

The robot's position on the digital map therefore acts like a **live moving marker**.

The actual warehouse obstacles (`1`) must never be overwritten by the robot position.

---

# 3. Real-World Coordinates

The occupancy grid should correspond to real-world warehouse coordinates.

For example:

```text
Warehouse = 10m × 6m
```

The warehouse can be divided into cells of a chosen resolution.

For example:

```text
Resolution = 10 cm/cell
```

Then:

```text
10m × 6m
=
100 × 60 cells
```

Each cell corresponds to a physical location in the warehouse.

The robot's position can therefore be represented both as:

```text
Grid coordinate:
(row, column)
```

and as:

```text
World coordinate:
(x, y, θ)
```

where `θ` represents the robot's orientation.

---

# 4. Warehouse Map Creation

The warehouse manager should be able to draw the physical warehouse layout on paper.

The drawing should contain:

- Warehouse dimensions
- Walls
- Shelves
- Shelf IDs
- Shelf locations
- Robot starting position
- Doors/restricted areas if required

Example:

```text
                 10 m
┌──────────────────────────────────┐
│                                  │
│   ┌──────────┐    ┌──────────┐   │
│   │   S01    │    │   S02    │   │
│   └──────────┘    └──────────┘   │
│                                  │
│             ┌──────────┐         │
│             │   S03    │         │
│             └──────────┘         │
│                                  │
│                    R             │
└──────────────────────────────────┘
                  6 m
```

The manager photographs/scans the drawing.

The system then converts it into a digital map.

---

# 5. Paper Map → Digital Map

The map processing pipeline should be:

```text
Paper Map
    ↓
Camera / Scanner
    ↓
Perspective Correction
    ↓
OpenCV
    ↓
Line / Rectangle Detection
    ↓
OCR
    ↓
Detect Shelf IDs
    ↓
Extract Dimensions
    ↓
Calculate Coordinates
    ↓
Generate Digital Warehouse Map
    ↓
User Reviews / Corrects Map
    ↓
Generate Navigation Map
```

Initially, use a **standardized drawing format** instead of trying to understand arbitrary hand-drawn maps.

---

# 6. Shelf Coordinates and Barcode Mapping

Every shelf should have a unique QR code or barcode.

For example:

```text
S01 → QR-S01
S02 → QR-S02
S03 → QR-S03
S04 → QR-S04
```

The system should maintain a mapping between:

```text
Shelf ID
      ↓
Barcode/QR code
      ↓
Shelf coordinates
```

For example:

```text
S01
QR-S01
Coordinates = (2.0, 4.0)
```

```text
S02
QR-S02
Coordinates = (7.0, 4.0)
```

```text
S03
QR-S03
Coordinates = (7.0, 2.0)
```

Therefore, the barcode is not only an identifier for the shelf. The backend knows the **coordinates associated with that barcode**.

---

# 7. Product → Barcode → Shelf Coordinates

The inventory database should maintain the relationship:

```text
Product
   ↓
Product Barcode
   ↓
Shelf Barcode
   ↓
Shelf Coordinates
```

For example:

```text
Banana
   ↓
Product Barcode: 890123456789
   ↓
Shelf Barcode: QR-S03
   ↓
Shelf Coordinates: (7.0, 2.0)
```

Therefore, when the user says:

> "Find Banana."

the system can perform:

```text
User says "Banana"
        ↓
Identify product
        ↓
Inventory Database
        ↓
Find Banana's product record
        ↓
Find Banana's assigned shelf
        ↓
Get shelf barcode
        ↓
Get shelf coordinates associated with that barcode
        ↓
Target = S03
        ↓
Target coordinates = (7.0, 2.0)
```

---

# 8. Robot Navigation

Once the target coordinates are known, the navigation system calculates a path through the `0` cells of the occupancy grid.

For example:

```text
2 0 0 0 0
1 1 1 0 1
0 0 0 0 1
0 1 1 1 1
0 0 0 0 0
```

The robot starts at `2`.

The target shelf is located at a known coordinate.

The path planner finds a route through free cells while avoiding `1` cells.

For the initial implementation, use:

```text
A*
```

The path is essentially:

```text
Current Robot Position
        ↓
Path Planner
        ↓
Find sequence of free cells
        ↓
Move through those cells
        ↓
Reach target shelf
```

---

# 9. Live Map Updating During Navigation

While the robot follows the path, its position on the occupancy grid must continuously update.

Suppose the path is:

```text
(2,2)
(2,3)
(2,4)
(3,4)
(4,4)
```

Initially:

```text
0 0 0 0 0
0 0 0 0 0
0 0 2 0 0
0 0 0 0 0
0 0 0 0 0
```

After moving to `(2,3)`:

```text
0 0 0 0 0
0 0 0 0 0
0 0 0 2 0
0 0 0 0 0
0 0 0 0 0
```

Then `(2,4)`:

```text
0 0 0 0 0
0 0 0 0 0
0 0 0 0 2
0 0 0 0 0
0 0 0 0 0
```

And so on.

The previous position returns to its underlying map value, while the new position becomes `2`.

This provides a continuously updated representation of the robot's location.

---

# 10. Robot Localization

The robot cannot simply assume that its position is correct forever.

Its estimated position should be continuously updated using sensors such as:

```text
Wheel Encoders
+
IMU
+
LiDAR
+
QR/Camera landmarks
```

Conceptually:

```text
Wheel Odometry ──┐
                 │
IMU ─────────────┼──→ Sensor Fusion → Current Robot Pose
                 │                    (x,y,θ)
LiDAR ───────────┤
                 │
QR landmarks ────┘
```

The localization system continuously determines:

```text
Current X
Current Y
Current Orientation
```

and updates the robot's position on the digital map.

---

# 11. QR/Barcode as Position Verification

When the robot approaches its target shelf:

```text
Navigation
    ↓
Approach S03 coordinates
    ↓
Camera scans QR
    ↓
Detect QR-S03
    ↓
Backend checks:
QR-S03 → S03 → expected coordinates
    ↓
Confirm correct shelf
```

If the robot expected:

```text
S03
```

but detected:

```text
QR-S02
```

it should not proceed with picking/placing.

Instead, it should recognize that it has reached the wrong location and perform a navigation/localization correction.

---

# 12. Product Detection at Shelf

After confirming the shelf:

```text
QR-S03
    ↓
Shelf confirmed
    ↓
Camera activated
    ↓
Object Detection
    ↓
Find requested product
```

For example:

```text
Shelf S03

┌───────────────────────────────┐
│ 🍎 🍌 🍌 🥤                  │
│ 🍎 🍌       🥤               │
└───────────────────────────────┘
```

If the requested product is Banana:

```text
Camera
   ↓
Object Detection
   ↓
Banana detected
   ↓
Determine product location
   ↓
Robotic Arm
```

---

# 13. Stocking Workflow

The complete product-stocking workflow should be:

```text
Manager gives product
        ↓
Manager says product name
        ↓
Identify product
        ↓
Query Inventory Database
        ↓
Does product already exist?
        │
   ┌────┴─────┐
   │          │
  YES         NO
   │          │
   ▼          ▼
Existing     Round-Robin
Shelf        Shelf Assignment
   │          │
   │          ▼
   │       Save new
   │       Product → Shelf
   │          │
   └────┬─────┘
        ▼
Find Shelf Barcode
        ↓
Find Shelf Coordinates
        ↓
Set Target Coordinate
        ↓
Path Planning
        ↓
Robot Navigation
        ↓
Continuously Update Robot Position
        ↓
Reach Target Area
        ↓
Scan Shelf QR/Barcode
        ↓
Confirm Correct Shelf
        ↓
Detect Product / Placement Area
        ↓
Robotic Arm
        ↓
PLACE
        ↓
Update Inventory Database
```

---

# 14. New Product Allocation — Round Robin

For the current version, if a product does not exist, use round-robin shelf assignment.

Suppose the shelves are:

```text
S01
S02
S03
S04
```

New products are assigned:

```text
Product 1 → S01
Product 2 → S02
Product 3 → S03
Product 4 → S04
Product 5 → S01
Product 6 → S02
```

The system maintains:

```text
next_shelf_index
```

and increments it after every new-product assignment.

Capacity optimization is **not part of the current implementation**.

---

# 15. Retrieval Workflow

When the user says:

> "Bring me a banana."

the system should perform:

```text
"Banana"
    ↓
Product Identification
    ↓
Inventory Database
    ↓
Find Banana
    ↓
Find Banana's assigned Shelf
    ↓
Get Shelf Barcode
    ↓
Get Shelf Coordinates
    ↓
Target = S03
    ↓
Path Planning
    ↓
Robot moves through free (0) cells
    ↓
Robot position continuously updates as 2
    ↓
Reach S03
    ↓
Scan QR-S03
    ↓
Confirm correct shelf
    ↓
Camera detects Banana
    ↓
Determine Banana position
    ↓
Robotic Arm picks Banana
    ↓
Robot returns to designated drop-off point
    ↓
Database quantity decreases
```

---

# 16. Complete Architecture

```text
                         USER / MANAGER
                               │
                               ▼
                       ┌─────────────────┐
                       │   BACKEND/API   │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ INVENTORY DB    │
                       │                 │
                       │ Products        │
                       │ Product Barcode │
                       │ Shelf IDs       │
                       │ Shelf Barcodes  │
                       │ Shelf Coordinates│
                       │ Quantities      │
                       └────────┬────────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
                   ▼                         ▼
             Inventory Logic            Digital Map
                   │                         │
          Existing / New?                    │
             /         \                     │
           YES          NO                   │
            │            │                   │
       Existing Shelf   Round Robin          │
            │            │                   │
            └─────┬──────┘                   │
                  ▼                          │
            Target Shelf                     │
                  │                          │
                  ▼                          │
          Shelf Barcode                     │
                  │                          │
                  ▼                          │
          Shelf Coordinates ─────────────────┘
                  │
                  ▼
             Path Planner
               A* / Nav2
                  │
                  ▼
             AUTONOMOUS ROBOT
                  │
       ┌──────────┼───────────┐
       │          │           │
       ▼          ▼           ▼
     LiDAR       IMU       Encoders
       │          │           │
       └──────────┼───────────┘
                  ▼
             Localization
                  │
                  ▼
          Current Robot Pose
               (x,y,θ)
                  │
                  ▼
        Update Occupancy Grid
                  │
             0 → free
             1 → obstacle
             2 → robot
                  │
                  ▼
             Robot Motors
                  │
                  ▼
          Target Shelf Area
                  │
                  ▼
           Camera / QR Scan
                  │
                  ▼
          Shelf Verification
                  │
                  ▼
          Object Detection
                  │
                  ▼
             Product Found
                  │
                  ▼
             Robotic Arm
                  │
             ┌────┴────┐
             ▼         ▼
            PICK      PLACE
             │         │
             └────┬────┘
                  ▼
           Update Inventory
```

---

# 17. Core Data Relationships

The important relationships in the system are:

```text
Product
   │
   ├── Product Name
   ├── Product Barcode
   │
   ▼
Shelf Assignment
   │
   ▼
Shelf ID
   │
   ▼
Shelf QR/Barcode
   │
   ▼
Shelf Coordinates
   │
   ▼
Navigation Map
   │
   ▼
Robot Target
```

For example:

```text
Banana
   ↓
Product Barcode = P890123
   ↓
Assigned Shelf = S03
   ↓
Shelf QR = QR-S03
   ↓
Coordinates = (7.0, 2.0)
   ↓
Navigation
   ↓
Robot reaches S03
   ↓
Scan QR-S03
   ↓
Confirm S03
   ↓
Find Banana
   ↓
Pick/Place
```

---

# 18. Core Principle

The system should maintain four separate but connected concepts:

### Inventory Database

> **What product is where?**

### Digital Warehouse Map

> **Where is that shelf physically located?**

### Localization

> **Where is the robot currently located?**

### Computer Vision

> **What is physically present at the target shelf?**

The overall process is:

```text
USER
 ↓
PRODUCT
 ↓
INVENTORY DATABASE
 ↓
PRODUCT → SHELF
 ↓
SHELF → BARCODE
 ↓
BARCODE → SHELF COORDINATES
 ↓
TARGET COORDINATE
 ↓
PATH PLANNING
 ↓
MOVE THROUGH FREE (0) CELLS
 ↓
CONTINUOUSLY UPDATE ROBOT POSITION (2)
 ↓
REACH TARGET
 ↓
SCAN SHELF BARCODE
 ↓
VERIFY LOCATION
 ↓
OBJECT DETECTION
 ↓
FIND PRODUCT
 ↓
ROBOTIC ARM
 ↓
PICK / PLACE
 ↓
UPDATE INVENTORY DATABASE
```

The central idea is that the **paper map creates the static warehouse/navigation map, the inventory database links products to shelf barcodes, the shelf barcodes are linked to shelf coordinates, and the robot dynamically updates its position on the navigation grid while moving toward the target shelf**.