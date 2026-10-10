"""Build the storeys of a photo-read building from its blocks, in code.

Given the reading's blocks (which storeys each spans) and intake's plans
(each tagged with its block), the storey list is fixed: Haiku, asked to
write it, modelled 3 or 4 storeys of a 14-storey reading in three of four
runs.
"""

from itertools import groupby

from loguru import logger

from src.agent.state import (
    PhotoReadingSchema,
    StoreySchema,
    StoreyZoneSchema,
    ZonePlanSchema,
)


def _runs(storeys: list[tuple[int, float, tuple[str, ...]]]):
    """Consecutive storeys with the same height and plans, as lists."""
    for _, run in groupby(storeys, key=lambda s: (s[1], s[2])):
        yield list(run)


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
        ValueError: If a plan names no block of the reading, or a block has
            no plan; intake can fix both.
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
    for run in _runs(storeys):
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
