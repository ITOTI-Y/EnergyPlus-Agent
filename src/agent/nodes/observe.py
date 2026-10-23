"""Read the building's photos or drawings once, before intake.

A forced tool call that must also write every specification leaves the
model no room to look at the images and break the building into blocks,
so the images are read in a call of their own. The reading is text for
intake, so revisions resend no image.
"""

import base64
import io
from pathlib import Path
from statistics import median_low
from typing import Any, Final, Literal, Self, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, model_validator

from src.agent import mesh_massing, segmentation
from src.agent._share import (
    IMAGE_TO_3D_API_KEY,
    IMAGE_TO_3D_URL,
    SEGMENT_URL,
    language_directive,
)
from src.agent.llm import create_vision_llm
from src.agent.nodes._share import structured
from src.agent.state import AgentState, AgentStateUpdate, PhotoReadingSchema


class TextContentPart(TypedDict):
    """LangChain multimodal text content part."""

    type: Literal["text"]
    text: str


class ImageContentPart(TypedDict):
    """LangChain multimodal image content part (base64-encoded)."""

    type: Literal["image"]
    source_type: Literal["base64"]
    mime_type: str
    data: str


type ContentPart = TextContentPart | ImageContentPart

IMAGE_TYPES: Final = (".png", ".jpg", ".jpeg", ".webp", ".gif")
"""Accepted file types; every image is sent as a JPEG."""

OBSERVE_SYSTEM_PROMPT: Final = """You read photos or drawings of a building for an energy model.
Report what the images show, and estimates with the cues behind them.
- Count storeys one by one from the roof down to the ground, one entry per
  storey in `storey_rows`. The repeated counts of an enlarged facade strip
  given with the images are the most reliable evidence; storeys the strip
  does not reach, and storeys hidden by other blocks or trees, count too.
- Decompose the massing into blocks: a podium, a tower on it, a service
  core with few or no windows, wings. A block has one rectangular
  footprint on all its storeys; where the footprint changes, split it into
  blocks. Account for every numbered volume boxed in the images, listing
  in each block's `volumes` the numbers of the volumes it is made of. Describe
  each block as it appears: where one rises out of
  another, code keeps the taller block whole and trims the lower one
  around it.
- Place each block with `x_m` and `front_setback_m` relative to the main
  (tallest) block at 0, 0; fronts face the camera's main facade.
- Estimate dimensions from scale cues: an office storey is about 3.5-4 m,
  a door about 2.2 m, a car about 4.5 m; count the facade bays, and
  compare widths and depths with the storey height. Give the ground storey's
  height and the other storeys' height: code stacks the storeys from them
  and from each block's `bottom_storey` and `storeys`.
- For each visible side of each block give the space behind it, the
  window type and an estimated window-to-wall ratio that fits both. Judge
  the space from the windows: a side with no windows, or a few small ones
  in a column, is a stair or service core (ratio at most 0.1); continuous
  bands are offices; storey-high glazing at the ground is a lobby or shop;
  louvres are plant. A solid face shows no ribbon, whatever the sides
  around it show. For separate windows give one window's width and height,
  measured against the storey height.
- List in `assumptions` what the images do not show (back facades,
  depth) and what you assumed for it.
- Write `reading` first, then the other fields consistently with it.
"""


MIN_LONG_SIDE_PX: Final = 1700
"""Images with a shorter long side are enlarged to it before reading.

Vision encoders commonly give one token to a 32 x 32 pixel patch,
so in a phone photo of ~800 px a punched window or a window band is a
fraction of a token, and the model undercounts storeys. Enlarging adds no
detail but spreads each row over several tokens.
"""


STRIP_COUNTS: Final = 3
"""Independent counts of a facade strip; the reading gets all of them.

Counting the rows of one enlarged strip is the most reliable storey count
a vision model of this size gives, but single counts scatter by a few
storeys either way, so the strip is counted several times.
"""

BOX_COORDINATES = """
Coordinates run from 0 to 1000 across the image's width and height, (0, 0)
at the top left and (1000, 1000) at the bottom right; they are not pixels."""

STRIP_PROMPT = (
    """Find the vertical strip of this building's facade where its
storeys are marked most completely and regularly from the roof down to the
ground: one window row, band or opening per storey, least hidden by other
parts of the building, trees or objects. Give its bounding box from the
roof line down to the ground."""
    + BOX_COORDINATES
)

COUNT_PROMPT = """This is an enlarged strip of a building's facade. Number its
storeys from the top of the strip down to its bottom, one entry per storey,
saying what marks it; a storey whose marker is hidden counts too."""


class BoxSchema(BaseModel):
    """A box in an image, in the 0-1000 coordinates vision models commonly ground in."""

    bbox_2d: list[int] = Field(
        min_length=4,
        max_length=4,
        description="x1, y1, x2, y2 in 0-1000 coordinates relative to the image "
        "width and height",
    )

    @model_validator(mode="after")
    def _is_a_box_inside_the_image(self) -> Self:
        x1, y1, x2, y2 = self.bbox_2d
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(
                f"bbox_2d {self.bbox_2d} must be x1 < x2 and y1 < y2 within 0-1000"
            )
        return self

    def pixels(self, image: Image.Image) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_2d
        w, h = image.size
        return x1 * w // 1000, y1 * h // 1000, x2 * w // 1000, y2 * h // 1000


class StoreyStripSchema(BoxSchema):
    """The facade strip to count storeys on."""


VOLUMES_PROMPT = (
    """Box every distinct volume of the building in this image:
each part whose faces stand in a different plane or rise to a different
height than its neighbours. One box per volume, covering the part of it
that is visible."""
    + BOX_COORDINATES
)


class VolumeSchema(BoxSchema):
    """One volume of the building as the image shows it."""

    label: str = Field(description="Short name, e.g. 'tower', 'base'")


class VolumesSchema(BaseModel):
    """The volumes of the building in one image."""

    volumes: list[VolumeSchema] = Field(min_length=1)


class StripCountSchema(BaseModel):
    """The storeys of one facade strip, top down."""

    storeys: list[str] = Field(
        min_length=1, description="One entry per storey: what marks it"
    )


def _open(path: str) -> Image.Image:
    """The image file in RGB, enlarged to MIN_LONG_SIDE_PX if smaller.

    Raises:
        ValueError: If the file type is not a supported image type.
    """
    p = Path(path)
    if p.suffix.lower() not in IMAGE_TYPES:
        raise ValueError(
            f"{p.name}: unsupported image type; use {', '.join(IMAGE_TYPES)}"
        )
    with Image.open(p) as image:
        # Phone photos are often stored sideways with an orientation tag.
        return _enlarged(ImageOps.exif_transpose(image).convert("RGB"))


def _enlarged(image: Image.Image) -> Image.Image:
    scale = MIN_LONG_SIDE_PX / max(image.size)
    if scale <= 1:
        return image
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)


def _image_part(image: Image.Image) -> ImageContentPart:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return ImageContentPart(
        type="image",
        source_type="base64",
        mime_type="image/jpeg",
        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def load_image_part(path: str) -> ImageContentPart:
    """An image file as a JPEG content part, enlarged if small.

    Raises:
        ValueError: If the file type is not a supported image type.
    """
    return _image_part(_open(path))


def _message(*parts: ContentPart) -> HumanMessage:
    return HumanMessage(content=cast("list[str | dict[str, Any]]", list(parts)))


def _text(text: str) -> TextContentPart:
    return TextContentPart(type="text", text=text)


def _up_to_the_roof(
    strip: StoreyStripSchema, volumes: list[VolumeSchema]
) -> StoreyStripSchema:
    """The strip run up to the top of the highest volume behind its middle.

    A strip boxed on its own often starts a few storeys below the roof; the
    volume it is part of, boxed separately, reaches it.
    """
    x1, y1, x2, y2 = strip.bbox_2d
    middle = (x1 + x2) / 2
    tops = [v.bbox_2d[1] for v in volumes if v.bbox_2d[0] <= middle <= v.bbox_2d[2]]
    return StoreyStripSchema(bbox_2d=[x1, min([y1, *tops]), x2, y2])


def _count_strip(
    llm: BaseChatModel, image: Image.Image, volumes: list[VolumeSchema]
) -> tuple[StoreyStripSchema, Image.Image, list[int]]:
    """Locate the best-marked facade strip, run it up to the roof, enlarge
    it and count it.

    The box uses the 0-1000 coordinates vision models commonly ground objects in.
    """
    located, _ = structured(
        llm,
        StoreyStripSchema,
        [_message(_text(STRIP_PROMPT), _image_part(image))],
        lambda s: s,
    )
    strip = _up_to_the_roof(located, volumes)
    enlarged = _enlarged(image.crop(strip.pixels(image)))
    ask: list[BaseMessage] = [_message(_text(COUNT_PROMPT), _image_part(enlarged))]
    counts = [
        len(structured(llm, StripCountSchema, ask, lambda c: c)[0].storeys)
        for _ in range(STRIP_COUNTS)
    ]
    return strip, enlarged, counts


def _boxed_volumes(
    llm: BaseChatModel, image: Image.Image, first: int
) -> tuple[list[VolumeSchema], Image.Image]:
    """The building's volumes, and a copy of the image with them boxed and
    numbered from ``first``.

    Asked on its own, the model sets apart volumes that a reading of the
    whole building merges, e.g. a part projecting in front of a slab.
    """
    reply, _ = structured(
        llm,
        VolumesSchema,
        [_message(_text(VOLUMES_PROMPT), _image_part(image))],
        lambda r: r,
    )
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    size = max(image.size) // 40
    for number, volume in enumerate(reply.volumes, start=first):
        x1, y1, x2, y2 = volume.pixels(image)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=size // 6 + 1)
        draw.text(
            (x1 + size // 2, y1 + size // 2), str(number), fill="red", font_size=size
        )
    return reply.volumes, boxed


BUILDING_MARGIN: Final = 0.03
"""Share of the image added around the boxed volumes when cropping to them."""


def _around(
    volumes: list[VolumeSchema], image: Image.Image
) -> tuple[int, int, int, int]:
    """Pixels of the box around all volumes, with BUILDING_MARGIN.

    An image-to-3D model builds whatever the image shows: neighbouring
    buildings, trees and wires around the target become volumes, and the
    target flattens into a facade. Cropped to the boxed volumes, it builds
    the target alone.
    """
    margin = BUILDING_MARGIN * 1000
    box = BoxSchema(
        bbox_2d=[
            max(0, round(min(v.bbox_2d[0] for v in volumes) - margin)),
            max(0, round(min(v.bbox_2d[1] for v in volumes) - margin)),
            min(1000, round(max(v.bbox_2d[2] for v in volumes) + margin)),
            min(1000, round(max(v.bbox_2d[3] for v in volumes) + margin)),
        ]
    )
    return box.pixels(image)


def observe_node(state: AgentState) -> AgentStateUpdate:
    """Read the images with the vision model; nothing to do without images.

    For each image the best-marked facade strip is located, enlarged and
    counted, and the building's volumes are boxed; the reading gets every
    image, each strip with its counts, and each image with its volumes
    boxed.

    Raises:
        ValueError: If images are given but no vision model is configured,
            before any LLM call.
    """
    if not state.image_paths or state.photo_reading is not None:
        return AgentStateUpdate()
    llm = create_vision_llm()
    images = [_open(path) for path in state.image_paths]
    parts: list[ContentPart] = [_text(state.user_input)]
    parts += [_image_part(image) for image in images]
    # Volumes are numbered across the images; blocks name them by number.
    numbered = 0
    for number, image in enumerate(images, start=1):
        first = numbered + 1
        volumes, boxed = _boxed_volumes(llm, image, first)
        if number == 1:
            first_volumes = volumes
        numbered += len(volumes)
        strip, enlarged, counts = _count_strip(llm, image, volumes)
        logger.info("photo {}: strip {} counted {}", number, strip.bbox_2d, counts)
        parts += [
            _text(
                f"The facade strip {strip.bbox_2d} (0-1000) of image {number}, "
                f"enlarged below, was counted {len(counts)} times independently, "
                f"each count covering the whole strip from its top to its "
                f"bottom: {counts} storeys, median {median_low(counts)}. Base "
                f"`storey_rows` on the median, adding only storeys the strip "
                f"does not reach."
            ),
            _image_part(enlarged),
        ]
        listed = "; ".join(f"{n} {v.label}" for n, v in enumerate(volumes, start=first))
        parts += [
            _text(
                f"The volumes of image {number}, boxed and numbered in the "
                f"image below: {listed}. Make each a block, or say in "
                f"`reading` which block it is part of."
            ),
            _image_part(boxed),
        ]
    system = SystemMessage(content=OBSERVE_SYSTEM_PROMPT + language_directive())
    _, reading = structured(
        llm,
        PhotoReadingSchema,
        [system, _message(*parts)],
        lambda r: r,
    )
    if IMAGE_TO_3D_URL:
        reading = _with_mesh_blocks(reading, images[0], first_volumes)
    return AgentStateUpdate(photo_reading=reading)


def _with_mesh_blocks(
    reading: PhotoReadingSchema, image: Image.Image, volumes: list[VolumeSchema]
) -> PhotoReadingSchema:
    """The reading with the blocks of a 3D model of the image's building.

    With SEGMENT_URL the building is cut out of the image, its volumes
    choosing it among the segmented buildings; without, the image is
    cropped to the volumes the reading's blocks are made of, leaving out
    neighbours the vision model boxed too. When no usable building comes
    out, because segmentation finds none, a server fails or the 3D model
    shows none, the reading keeps its own blocks and an assumption says why.
    """
    used = {v for b in reading.blocks for v in b.volumes if 1 <= v <= len(volumes)}
    building = [v for n, v in enumerate(volumes, start=1) if n in used] or volumes
    try:
        if SEGMENT_URL:
            target = segmentation.silhouette(
                segmentation.segment(image, "building", SEGMENT_URL),
                [v.pixels(image) for v in building],
            )
            shown = segmentation.cut_out(image, target)
        else:
            shown = image.crop(_around(building, image))
        mesh = mesh_massing.generate_mesh(shown, IMAGE_TO_3D_URL, IMAGE_TO_3D_API_KEY)
        meshed = mesh_massing.with_mesh_blocks(reading, mesh)
    # The vision model's reading is complete by now; the 3D model only
    # refines its blocks. OSError covers unreachable or failing servers.
    except (mesh_massing.UnusableMeshError, ValueError, OSError) as e:
        logger.warning("3D model not used: {}", e)
        return reading.model_copy(
            update={"assumptions": [*reading.assumptions, f"No 3D model blocks: {e}."]}
        )
    logger.info(
        "blocks from the 3D model: {}",
        [(b.name, b.width_m, b.depth_m, b.front_setback_m) for b in meshed.blocks],
    )
    return meshed
