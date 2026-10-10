"""Split a rectangular floor plan into perimeter zones and a core zone.

The DOE commercial prototypes zone each office floor as four perimeter
zones 15 ft (4.57 m) deep along the exterior walls and one core zone
(medium office technical support document, appendix 4A). The split is a
fixed rule, so code does it rather than the LLM writing the corners.
"""

from collections.abc import Sequence
from typing import Final

PERIMETER_DEPTH_M: Final = 4.57
"""Depth of a perimeter zone, as in the DOE prototypes (15 ft)."""

MIN_CORE_WIDTH_M: Final = 3.0
"""Narrowest core left between two opposite perimeter zones."""

_TOLERANCE: Final = 1e-6

type Point2 = tuple[float, float]


def _facing(outward: Point2) -> str:
    """Compass side of an outward normal in model axes (y north, x east)."""
    x, y = outward
    if abs(y) >= abs(x):
        return "N" if y > 0 else "S"
    return "E" if x > 0 else "W"


def perimeter_core(
    corners: Sequence[Point2], depth: float = PERIMETER_DEPTH_M
) -> list[tuple[str, list[Point2]]]:
    """Four perimeter zones ``depth`` deep and a core zone of a rectangle.

    Each perimeter zone is the trapezoid between one side and the core, and
    is named by the compass side it faces ('N', 'E', 'S', 'W'; 'P1'-'P4'
    for a rectangle turned exactly 45 degrees); the core is 'Core'. Corners
    come back counterclockwise.

    Raises:
        ValueError: If the plan is not a rectangle, or a side is too short
            to leave a core of MIN_CORE_WIDTH_M between perimeter zones.
    """
    if len(corners) != 4:
        raise ValueError(
            f"perimeter_core needs a rectangle, got {len(corners)} corners"
        )
    points = list(corners)
    area2 = sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True)
    )
    if area2 < 0:
        points.reverse()
    sides = []
    for i in range(4):
        (x0, y0), (x1, y1) = points[i], points[(i + 1) % 4]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if length <= _TOLERANCE:
            raise ValueError("perimeter_core needs a rectangle, got a repeated corner")
        sides.append(((x1 - x0) / length, (y1 - y0) / length, length))
    for (ux, uy, _), (vx, vy, _) in zip(sides, sides[1:] + sides[:1], strict=True):
        if abs(ux * vx + uy * vy) > 1e-3:
            raise ValueError(
                "perimeter_core needs a rectangle: corners are not right angles"
            )
    shortest = min(length for *_, length in sides)
    if shortest < 2 * depth + MIN_CORE_WIDTH_M:
        raise ValueError(
            f"a {shortest:.1f} m side is too short for perimeter_core (needs "
            f"{2 * depth + MIN_CORE_WIDTH_M:.2f} m); use zoning 'single'"
        )
    # Inward normal of a counterclockwise side is its direction turned left.
    inward = [(-uy, ux) for ux, uy, _ in sides]
    inner = [
        (
            x + depth * (inward[i][0] + inward[i - 1][0]),
            y + depth * (inward[i][1] + inward[i - 1][1]),
        )
        for i, (x, y) in enumerate(points)
    ]
    labels = [_facing((-nx, -ny)) for nx, ny in inward]
    if len(set(labels)) < 4:
        labels = [f"P{i + 1}" for i in range(4)]
    zones = [
        (labels[i], [points[i], points[(i + 1) % 4], inner[(i + 1) % 4], inner[i]])
        for i in range(4)
    ]
    return [*zones, ("Core", inner)]
