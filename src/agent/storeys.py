"""Build the storeys of a photo-read building from its blocks, in code.

Given the reading's blocks (which storeys each spans) and intake's plans
(each tagged with its block), the storey list is fixed, so code builds it
rather than asking the LLM to write it.
"""

from itertools import groupby
from typing import Final

from loguru import logger
from shapely import Polygon, box, unary_union

from src.agent.state import (
    PhotoReadingSchema,
    StoreySchema,
    StoreyZoneSchema,
    ZonePlanSchema,
)

FOOTPRINT_TOLERANCE_M: Final = 0.05
"""Plan corners may lie this far outside their block's footprint."""

FOOTPRINT_AREA_TOLERANCE: Final = 0.02
"""Share of a block's footprint its plans may leave uncovered."""


def _check_footprints(plans: list[ZonePlanSchema], reading: PhotoReadingSchema) -> None:
    """Each block's plans lie in its footprint and together fill it.

    The reading fixes where each block stands, setbacks included; intake
    only chooses how to divide a block into plans.
    """
    problems = []
    for name, (x0, y0, x1, y1) in reading.footprints().items():
        footprint = box(x0, y0, x1, y1)
        shapes = {
            p.key: Polygon([(q.x, q.y) for q in p.plan])
            for p in plans
            if p.block == name
        }
        # A plan with its corners out of order crosses itself; the union
        # below fails on it.
        crossed = sorted(key for key, shape in shapes.items() if not shape.is_valid)
        if crossed:
            problems.append(
                f"plans {crossed} of block '{name}' cross themselves: list "
                f"their corners in order around the zone"
            )
            continue
        outside = sorted(
            key
            for key, shape in shapes.items()
            if not footprint.buffer(FOOTPRINT_TOLERANCE_M).contains(shape)
        )
        covered = unary_union(list(shapes.values())).area / footprint.area
        if outside or covered < 1 - FOOTPRINT_AREA_TOLERANCE:
            problems.append(
                f"block '{name}' stands on x {x0}..{x1}, y {y0}..{y1}"
                + (f"; plans {outside} reach outside it" if outside else "")
                + f"; its plans cover {covered:.0%} of it"
            )
    if problems:
        raise ValueError(
            "each block's plans must exactly fill the block's footprint given "
            "with the photo reading: " + "; ".join(problems)
        )


def storeys_from_reading(
    plans: list[ZonePlanSchema], reading: PhotoReadingSchema
) -> list[StoreySchema]:
    """Storeys 1 up to the top of the highest block, plans by block.

    Storey k holds the plans of the blocks spanning it, the ground storey
    ``reading.ground_storey_height_m`` high and the others
    ``reading.storey_height_m``. Consecutive storeys with the same plans
    and height form a run; a run of three or more is modelled as its first
    storey, one typical storey with a multiplier for the middle ones, and
    its last storey, as the DOE prototypes model ground, typical and top
    floors. Storeys are named 'S<k>' by their first storey.

    A storey no block spans takes the plans of the storey below it.

    Raises:
        ValueError: If a plan names no block of the reading, a block has no
            plan, or a block's plans do not exactly fill its footprint;
            intake can fix all three.
    """
    blocks = {b.name: b for b in reading.blocks}
    unknown = sorted(f"{p.key} ({p.block!r})" for p in plans if p.block not in blocks)
    if unknown:
        raise ValueError(
            f"with a photo reading every plan sets `block` to one of "
            f"{sorted(blocks)}; these do not: {unknown}"
        )
    unused = sorted(set(blocks) - {p.block for p in plans})
    if unused:
        raise ValueError(f"blocks of the photo reading without a plan: {unused}")
    _check_footprints(plans, reading)
    top = max(b.bottom_storey + b.storeys - 1 for b in reading.blocks)
    storeys = [
        (
            k,
            reading.ground_storey_height_m if k == 1 else reading.storey_height_m,
            tuple(
                p.key
                for p in plans
                if (b := blocks[p.block or ""]).bottom_storey
                <= k
                < b.bottom_storey + b.storeys
            ),
        )
        for k in range(1, top + 1)
    ]
    if empty := [k for k, _, keys in storeys if not keys]:
        # A gap is the reading's, which intake cannot change: the storeys
        # take the plans of the storey below (above, at the bottom).
        logger.warning("photo reading: no block spans storeys {}", empty)
        filled = [keys for _, _, keys in storeys]
        for i in range(1, len(filled)):
            filled[i] = filled[i] or filled[i - 1]
        for i in reversed(range(len(filled) - 1)):
            filled[i] = filled[i] or filled[i + 1]
        storeys = [
            (k, h, keys) for (k, h, _), keys in zip(storeys, filled, strict=True)
        ]
    result = []
    for _, group in groupby(storeys, key=lambda s: (s[1], s[2])):
        run = list(group)
        if len(run) >= 3:
            first, *middle, last = run
            groups = [(first, 1), (middle[0], len(middle)), (last, 1)]
        else:
            groups = [(storey, 1) for storey in run]
        result += [
            StoreySchema(
                name=f"S{k}",
                height=height,
                multiplier=count,
                zones=[StoreyZoneSchema(plan=key) for key in keys],
            )
            for (k, height, keys), count in groups
        ]
    return result
