"""Massing blocks of a photographed building from an image-to-3D mesh.

A vision model counts storeys and reads facades well, but sums up a
stepped podium as one or two boxes. An image-to-3D model (HY-3D)
recovers the setbacks; code takes each storey's plan from the mesh's
surface points in the storey's middle half and covers it with rectangles,
a rectangle repeated on consecutive storeys becoming one block.

The mesh comes from TokenHub's HY-3D API at ``IMAGE_TO_3D_URL``: glTF, Y
up, standing on y = 0. Its scale is arbitrary; the counted storeys and
their heights set it.
"""

import base64
import io
import json
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
import trimesh
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from shapely import affinity, contains_xy
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from src.agent.state import MassingBlockSchema, PhotoReadingSchema, Rectangle

MODEL: Final = "hy-3d-3.1"

FACE_COUNT: Final = 20_000

REQUEST_TIMEOUT_S: Final = 60
GENERATE_TIMEOUT_S: Final = 900
POLL_INTERVAL_S: Final = 5.0

SAMPLES: Final = 400_000
"""Surface points taken from a mesh: a few thousand per storey."""

GRID_M: Final = 1.0
"""Cell of a storey's plan grid, in metres."""

CLOSING_CELLS: Final = 2
"""Gaps in a wall up to this many cells (balconies, windows) are closed."""

MIN_SECTION_CELLS: Final = 15
"""Smaller footprint pieces, in cells, are poles, signs or specks."""

PROFILE_BANDS: Final = 100
PROFILE_CELLS: Final = 100
"""Bands up the height and cells across, for the unscaled profile."""
COVER: Final = 0.9
"""Share of a storey's section its rectangles cover."""

MIN_SIDE_M: Final = 4.0
MIN_BLOCK_AREA_M2: Final = 60.0
GROW: Final = 0.9
"""Share of a block the storey below must cover for the block to reach it."""

BOX_FILL: Final = 0.97
"""A mesh whose every section fills more of its bounding box than this is
the box an image-to-3D model returns when it fails on an image. A building
that is one box loses nothing by it: the reading's block is that box."""


class UnusableMeshError(ValueError):
    """The 3D model shows no building that blocks can be taken from."""


@dataclass
class _Block:
    rect: Rectangle
    bottom: int
    top: int


def _call(url: str, path: str, api_key: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/{path}",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return json.load(response)


def generate_mesh(image: Image.Image, url: str, api_key: str) -> trimesh.Trimesh:
    """The untextured mesh of an image from TokenHub's HY-3D API.

    Raises:
        urllib.error.URLError: If the API cannot be reached or refuses a call.
        TimeoutError: If the job does not complete in GENERATE_TIMEOUT_S.
        ValueError: If the job fails or its result holds no mesh.
    """
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    job = _call(
        url,
        "submit",
        api_key,
        {
            "model": MODEL,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "generate_type": "Geometry",
            "face_count": FACE_COUNT,
        },
    )
    job_id = job["id"]
    deadline = time.monotonic() + GENERATE_TIMEOUT_S
    while job["status"] not in {"completed", "failed"}:
        if time.monotonic() > deadline:
            raise TimeoutError(f"3D job {job_id} not done in {GENERATE_TIMEOUT_S} s")
        time.sleep(POLL_INTERVAL_S)
        job = _call(url, "query", api_key, {"model": MODEL, "id": job_id})
    glb_url = next(
        (d["url"] for d in job.get("data") or [] if d["type"] == "glb"), None
    )
    if job["status"] == "failed" or glb_url is None:
        raise ValueError(f"3D job {job_id} gave no mesh: {json.dumps(job)[:300]}")
    with urllib.request.urlopen(glb_url, timeout=REQUEST_TIMEOUT_S) as response:
        glb = response.read()
    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"3D job {job_id} returned no mesh for the image")
    return mesh


def _surface_points(mesh: trimesh.Trimesh) -> tuple[NDArray, NDArray]:
    """Points spread over the mesh's surface: plan positions, x east and y
    north (the mesh's -z), and heights above its lowest point."""
    points, _ = trimesh.sample.sample_surface(mesh, SAMPLES, seed=0)
    plan = np.column_stack([points[:, 0], -points[:, 2]])
    return plan, points[:, 1] - mesh.bounds[0][1]


def _without_specks(grid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    labels, count = ndimage.label(grid)
    sizes = ndimage.sum(grid, labels, range(1, count + 1))
    return np.isin(labels, 1 + np.flatnonzero(sizes >= MIN_SECTION_CELLS))


def _footprint(plan: NDArray, cell: float) -> Polygon | MultiPolygon:
    """The solid plan that surface points outline, on a grid of ``cell``.

    Image-to-3D meshes are often open shells, walls cut by balconies and
    windows, whose sections are not closed outlines: the cells the points
    fall in are joined across gaps of up to CLOSING_CELLS and filled
    inside; pieces under MIN_SECTION_CELLS cells are dropped.
    """
    if not len(plan):
        return Polygon()
    origin = plan.min(axis=0) - CLOSING_CELLS * cell
    cells = np.floor((plan - origin) / cell).astype(int)
    grid = np.zeros(cells.max(axis=0) + CLOSING_CELLS + 2, bool)
    grid[cells[:, 0], cells[:, 1]] = True
    size = 2 * CLOSING_CELLS + 1
    grid = ndimage.binary_closing(grid, np.ones((size, size), bool))
    # Specks go first, so that they wall no cell in.
    grid = _without_specks(grid)
    grid = _without_specks(ndimage.binary_fill_holes(grid) | (_walled_sides(grid) >= 3))
    boxes = []
    for i in range(grid.shape[0]):
        runs = np.flatnonzero(np.diff(np.concatenate([[0], grid[i], [0]]).astype(int)))
        for j0, j1 in zip(runs[::2], runs[1::2], strict=True):
            x0, y0 = origin + (i, j0) * np.array([cell, cell])
            boxes.append(box(x0, y0, x0 + cell, y0 + (j1 - j0) * cell))
    return unary_union(boxes)


def _walled_sides(walls: NDArray) -> NDArray:
    """For each cell, on how many of its four sides a wall cell lies further
    along its row or column.

    A cell walled on three sides is inside an outline open on the fourth,
    as a wall cut by a wide gap leaves it; a notch of an L-shaped plan is
    walled on two.
    """
    sides = np.zeros(walls.shape, int)
    for axis in (0, 1):
        before = np.maximum.accumulate(walls, axis=axis)
        after = np.flip(np.maximum.accumulate(np.flip(walls, axis), axis=axis), axis)
        sides += before.astype(int) + after.astype(int)
    return sides


def _band(plan: NDArray, heights: NDArray, low: float, high: float) -> NDArray:
    return plan[(heights >= low) & (heights < high)]


def _profile(plan: NDArray, heights: NDArray) -> tuple[NDArray, NDArray]:
    """Footprint areas of thin bands up the height, as shares of the plan's
    bounding box, and the bands' mid-heights."""
    top = heights.max()
    edges = np.linspace(0, top, PROFILE_BANDS + 1)
    extent = (plan.max(axis=0) - plan.min(axis=0)).max()
    box_area = float(np.prod(plan.max(axis=0) - plan.min(axis=0)))
    cell = extent / PROFILE_CELLS
    areas = np.array(
        [
            _footprint(_band(plan, heights, lo, hi), cell).area
            for lo, hi in pairwise(edges)
        ]
    )
    return areas / box_area, (edges[:-1] + edges[1:]) / 2


def _fills_its_box(profile: NDArray, middles: NDArray) -> bool:
    """Whether every band from 10% to 90% of the height fills BOX_FILL of
    the plan's bounding box."""
    share = middles / middles.max()
    return bool(np.all(profile[(share > 0.1) & (share < 0.9)] >= BOX_FILL))


def roof_height(profile: NDArray, middles: NDArray) -> float:
    """The highest band whose footprint holds at least half the slab's.

    The slab's footprint is the median between 55% and 90% of the height;
    antennas and rooftop plant above the roof hold far less.
    """
    share = middles / middles.max()
    slab = np.median(profile[(share > 0.55) & (share < 0.9)])
    return float(middles[profile >= 0.5 * slab].max())


def _turn(plan: NDArray, angle: float) -> NDArray:
    a = np.radians(-angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return plan @ rotation.T


def _front_angle(top: Polygon | MultiPolygon, wide: bool) -> float:
    """The turn that lines the top storey's walls up with the axes, its
    front facing south.

    The walls give the turn up to quarter turns. Those within a quarter of
    the camera's view keep a facade the photo shows at the front: when there
    are two, the one making the plan wide or deep as ``wide`` says wins, as
    a corner view leaves it open.
    """
    corners = list(top.minimum_rotated_rectangle.exterior.coords)
    (ax, ay), (bx, by) = corners[0], corners[1]
    angle = (np.degrees(np.arctan2(by - ay, bx - ax)) + 45) % 90 - 45
    candidates = [angle] if abs(angle) < 1 else [angle, angle - 90 * np.sign(angle)]

    def matches(turn: float) -> bool:
        x0, y0, x1, y1 = affinity.rotate(top, -turn, origin=(0, 0)).bounds
        return (x1 - x0 >= y1 - y0) == wide

    return next((c for c in candidates if matches(c)), angle)


def _rectangles(plan: Polygon | MultiPolygon) -> list[Rectangle]:
    """Axis-aligned rectangles covering COVER of a plan, largest first."""
    if plan.is_empty:
        return []
    x0, y0, x1, y1 = plan.bounds
    xs = np.arange(x0, x1, GRID_M) + GRID_M / 2
    ys = np.arange(y0, y1, GRID_M) + GRID_M / 2
    grid = contains_xy(plan, *np.meshgrid(xs, ys))  # rows: y, columns: x
    total = grid.sum()
    found: list[Rectangle] = []
    min_cells = MIN_SIDE_M / GRID_M
    while grid.sum() > (1 - COVER) * total:
        best, best_area = None, 0
        heights = np.zeros(grid.shape[1], int)
        # Largest rectangle of free cells: the histogram method row by row.
        for row in range(grid.shape[0]):
            heights = np.where(grid[row], heights + 1, 0)
            stack: list[tuple[int, int]] = []
            for col in range(len(heights) + 1):
                height = heights[col] if col < len(heights) else 0
                start = col
                while stack and stack[-1][1] >= height:
                    start, h = stack.pop()
                    width = col - start
                    if width * h > best_area and min(width, h) >= min_cells:
                        best_area, best = width * h, (row - h + 1, row + 1, start, col)
                stack.append((start, height))
        if best is None:
            break
        r0, r1, c0, c1 = best
        grid[r0:r1, c0:c1] = False
        found.append(
            (x0 + c0 * GRID_M, y0 + r0 * GRID_M, x0 + c1 * GRID_M, y0 + r1 * GRID_M)
        )
    return found


def _iou(a: Rectangle, b: Rectangle) -> float:
    return box(*a).intersection(box(*b)).area / box(*a).union(box(*b)).area


def blocks_from_mesh(
    mesh: trimesh.Trimesh, reading: PhotoReadingSchema
) -> list[MassingBlockSchema]:
    """The mesh's massing as blocks on the reading's storeys.

    The mesh is scaled so its roof is at the top of the counted storeys,
    each storey's plan taken from the surface points in its middle half, and
    turned so the top storey's walls run along the axes, a facade the photo
    shows at the front (see ``_front_angle``). From the top down, a block reaches the
    storey below while that storey covers it; what a storey adds beyond the
    blocks above becomes new blocks. The highest, largest block is the main
    one; the others are placed relative to its west end and front.

    Raises:
        UnusableMeshError: If the mesh is a filled box, or shows no part of
            the top storey large enough to model.
    """
    plan, heights = _surface_points(mesh)
    profile, middles = _profile(plan, heights)
    if _fills_its_box(profile, middles):
        raise UnusableMeshError(
            "the 3D model is a filled box, as the image-to-3D model returns "
            "when it fails on an image"
        )
    storeys = len(reading.storey_rows)
    ground, typical = reading.ground_storey_height_m, reading.storey_height_m
    scale = (ground + (storeys - 1) * typical) / roof_height(profile, middles)
    plan, heights = plan * scale, heights * scale
    # The middle half of each storey: floors and ledges stay out.
    bands = [(ground * 0.25, ground * 0.75)] + [
        (ground + (k - 2 + 0.25) * typical, ground + (k - 2 + 0.75) * typical)
        for k in range(2, storeys + 1)
    ]
    top = _footprint(_band(plan, heights, *bands[-1]), GRID_M)
    if top.is_empty:
        raise UnusableMeshError("the 3D model shows no building on the top storey")
    # The largest read block shows the building's proportions; the highest
    # may be a narrow core.
    largest = max(reading.blocks, key=lambda b: b.width_m * b.depth_m * b.storeys)
    plan = _turn(plan, _front_angle(top, largest.width_m >= largest.depth_m))
    plans = [_footprint(_band(plan, heights, *b), GRID_M) for b in bands]

    blocks: list[_Block] = []
    for k in range(storeys, 0, -1):
        plan = plans[k - 1]
        for b in blocks:
            if (
                b.bottom == k + 1
                and plan.intersection(box(*b.rect)).area >= GROW * box(*b.rect).area
            ):
                b.bottom = k
        held = unary_union([box(*b.rect) for b in blocks if b.bottom == k])
        rest = plan if held.is_empty else plan.difference(held)
        for r in _rectangles(rest):
            if (r[2] - r[0]) * (r[3] - r[1]) < MIN_BLOCK_AREA_M2:
                continue
            # The same rectangle as a block ending just above continues it.
            above = next(
                (b for b in blocks if b.bottom == k + 1 and _iou(b.rect, r) >= GROW),
                None,
            )
            if above is None:
                blocks.append(_Block(r, k, k))
            else:
                above.bottom = k
    main = max(
        blocks,
        key=lambda b: (b.top, (b.rect[2] - b.rect[0]) * (b.rect[3] - b.rect[1])),
        default=None,
    )
    if main is None or main.top != storeys:
        raise UnusableMeshError(
            f"the 3D model's top storey has no part of at least "
            f"{MIN_BLOCK_AREA_M2} m2 to model"
        )
    x0, y0 = main.rect[0], main.rect[1]
    return [
        MassingBlockSchema(
            name=f"massing {n}",
            role="tower" if b is main else "podium",
            storeys=b.top - b.bottom + 1,
            bottom_storey=b.bottom,
            width_m=round(b.rect[2] - b.rect[0], 1),
            depth_m=round(b.rect[3] - b.rect[1], 1),
            x_m=round(b.rect[0] - x0, 1),
            front_setback_m=round(b.rect[1] - y0, 1),
            volumes=[],
        )
        for n, b in enumerate(blocks, start=1)
    ]


END_SHARE: Final = 0.1
"""Share of the building's width within which read blocks end together."""

type Extent = tuple[tuple[float, float], tuple[float, float], tuple[int, int]]
"""West-east and front-back shares of the building, and first and last storey."""


def _extents(blocks: list[MassingBlockSchema]) -> dict[str, Extent]:
    """Each block's extent, its plan spread over all the blocks' plan."""
    west = min(b.x_m for b in blocks)
    width = max(b.x_m + b.width_m for b in blocks) - west
    front = min(b.front_setback_m for b in blocks)
    depth = max(b.front_setback_m + b.depth_m for b in blocks) - front
    return {
        b.name: (
            ((b.x_m - west) / width, (b.x_m + b.width_m - west) / width),
            (
                (b.front_setback_m - front) / depth,
                (b.front_setback_m + b.depth_m - front) / depth,
            ),
            (b.bottom_storey, b.bottom_storey + b.storeys),
        )
        for b in blocks
    }


def _iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    union = max(a[1], b[1]) - min(a[0], b[0])
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / union if union else 0.0


def _match(
    extent: Extent, candidates: list[MassingBlockSchema], extents: dict[str, Extent]
) -> MassingBlockSchema | None:
    """The candidate most like the block: the product of their overlaps
    across the front, front to back and in storeys, each as an IoU.

    A read block's placement is a rough estimate but keeps the order of
    the blocks, which spreading both sets over the building preserves.
    """

    def likeness(c: MassingBlockSchema) -> float:
        other = extents[c.name]
        return (
            _iou_1d(extent[0], other[0])
            * _iou_1d(extent[1], other[1])
            * _iou_1d(extent[2], other[2])
        )

    best = max(candidates, key=likeness)
    return best if likeness(best) > 0 else None


def _end(
    extent: Extent,
    side: str,
    reading: PhotoReadingSchema,
    extents: dict[str, Extent],
) -> MassingBlockSchema | None:
    """The read block whose ``side`` ('left' west, 'right' east) facade the
    block's end shows: among those sharing its storeys and read on that
    side, the one most alike in storeys of those ending within END_SHARE of
    the building's width from it, else the one ending nearest. A podium
    may end with a tower; the tower's end shows over most storeys."""
    read = {f.block for f in reading.facades if f.side == side}
    candidates = [
        b
        for b in reading.blocks
        if b.name in read and _iou_1d(extent[2], extents[b.name][2]) > 0
    ]
    if not candidates:
        return None
    edge = 0 if side == "left" else 1

    def rank(block: MassingBlockSchema) -> tuple[bool, float, float]:
        other = extents[block.name]
        distance = abs(other[0][edge] - extent[0][edge])
        return distance > END_SHARE, -_iou_1d(extent[2], other[2]), distance

    return min(candidates, key=rank)


def with_mesh_blocks(
    reading: PhotoReadingSchema, mesh: trimesh.Trimesh
) -> PhotoReadingSchema:
    """The reading with its blocks replaced by the mesh's.

    The storeys and their heights stay the vision model's. A mesh block
    takes its name and its front and back from the read block most like it
    in place and storeys (see ``_match``), or the reading's main block when
    none is, numbered from the second mesh block taking the same name; its
    west and east ends from the read block ending there (see ``_end``): a
    mesh block may join read blocks, e.g. a slab and a core at its end.

    Raises:
        UnusableMeshError: As ``blocks_from_mesh``.
    """
    blocks = blocks_from_mesh(mesh, reading)
    read_main = max(
        reading.blocks, key=lambda b: (b.bottom_storey + b.storeys, b.width_m)
    )
    read_extents, mesh_extents = _extents(reading.blocks), _extents(blocks)
    facades = []
    named = []
    taken: Counter[str] = Counter()
    for block in blocks:
        extent = mesh_extents[block.name]
        whole = _match(extent, reading.blocks, read_extents) or read_main
        taken[whole.name] += 1
        name = whole.name + (f" {taken[whole.name]}" if taken[whole.name] > 1 else "")
        named.append(block.model_copy(update={"name": name}))
        for side in ("front", "back", "left", "right"):
            source = whole
            if side in ("left", "right"):
                source = _end(extent, side, reading, read_extents) or whole
            facades += [
                f.model_copy(update={"block": name})
                for f in reading.facades
                if f.block == source.name and f.side == side
            ]
    return PhotoReadingSchema.model_validate(
        reading.model_dump()
        | {
            "blocks": [b.model_dump() for b in named],
            "facades": [f.model_dump() for f in facades],
            "assumptions": [
                *reading.assumptions,
                "Blocks and setbacks are taken from a 3D model generated from the "
                "first image; its hidden sides are the model's guess.",
            ],
        }
    )
