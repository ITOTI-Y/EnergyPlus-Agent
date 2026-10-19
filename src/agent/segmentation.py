"""The photographed building cut out of its photo, for image-to-3D.

An image-to-3D model builds whatever the image shows, so neighbouring
buildings beside the target become part of its mesh. SAM 3 (a server at
SEGMENT_URL, see scripts/sam3_segment_server.py) finds every building in
the image; code keeps the one with the most pixels inside the boxes the
vision model drew around the target's volumes and fills each image column
between the target's top and its base, so that trees and poles in front of
it stay part of it: cut out, they would leave holes the 3D model builds as
gaps.
"""

import base64
import io
import json
import urllib.request
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

SEGMENT_TIMEOUT_S: Final = 120
BASE_SPAN: Final = 0.25
"""Share of the image's width over which a column takes its neighbours' base:
a tree hiding the building down to the street ends its own column early."""


def segment(image: Image.Image, prompt: str, url: str) -> list[NDArray[np.bool_]]:
    """Masks of every instance of ``prompt`` (e.g. 'building') in the image.

    Raises:
        urllib.error.URLError: If the server cannot be reached or fails.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    body = {
        "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "prompt": prompt,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/segment",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=SEGMENT_TIMEOUT_S) as response:
        instances = json.load(response)
    return [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(i["mask"])))) > 0
        for i in instances
    ]


def silhouette(
    buildings: list[NDArray[np.bool_]], boxes: list[tuple[int, int, int, int]]
) -> NDArray[np.bool_]:
    """The target building's pixels: the instance with the most pixels
    inside ``boxes``, filled per column from top to base, less the other
    buildings.

    ``boxes`` are the target's volumes in pixels (left, top, right, bottom).
    Boxes may be wide, e.g. around a canopy, and hold most of a neighbour,
    so only the one instance is kept.

    Raises:
        ValueError: If no building lies inside the boxes.
    """
    height, width = buildings[0].shape if buildings else (0, 0)
    boxed = np.zeros((height, width), bool)
    for left, top, right, bottom in boxes:
        boxed[top:bottom, left:right] = True
    inside = [int((mask & boxed).sum()) for mask in buildings]
    if max(inside, default=0) == 0:
        raise ValueError("no building found inside the boxed volumes")
    chosen = int(np.argmax(inside))
    target = buildings[chosen]
    others = np.zeros((height, width), bool)
    for k, mask in enumerate(buildings):
        if k != chosen:
            others |= mask
    columns = target.any(0)
    tops = np.where(columns, target.argmax(0), height)
    bases = np.where(columns, height - 1 - target[::-1].argmax(0), -1)
    span = max(1, round(BASE_SPAN * width))
    bases = np.where(columns, ndimage.maximum_filter1d(bases, size=span), -1)
    rows = np.arange(height)[:, None]
    filled = (rows >= tops) & (rows <= bases) & ~(others & ~target)
    return ndimage.binary_fill_holes(filled)


def cut_out(image: Image.Image, mask: NDArray[np.bool_]) -> Image.Image:
    """The image cropped to the mask, transparent outside it."""
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    rows, columns = np.nonzero(mask)
    return rgba.crop(
        (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        )
    )
