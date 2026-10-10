"""Read the building's photos or drawings once, before intake.

A forced tool call that must also write every specification gives the
model no room to look: with a short brief and a photo of a tower on a
podium, intake built one box of 11 storeys; reading the same photo in its
own call gave the podium, the tower and the core. The reading is text for
intake, so revisions resend no image.
"""

import base64
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent._share import language_directive
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


ContentPart = TextContentPart | ImageContentPart

IMAGE_TYPES: Final = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

OBSERVE_SYSTEM_PROMPT = """You read photos or drawings of a building for an energy model.
Report what the images show, and estimates with the cues behind them.
- Count the storeys of each block from its window bands; count the ground
  storey and any podium storeys separately.
- Decompose the massing into blocks: a podium, a tower on it, a service
  core with few or no windows, wings. A tall block on a low wide one is a
  tower on a podium.
- Estimate dimensions from scale cues: an office storey is about 3.5-4 m,
  a door about 2.2 m, a car about 4.5 m; count the facade bays. Give the
  ground storey's height and the other storeys' height: code stacks the
  storeys from them and from each block's `bottom_storey` and `storeys`.
- For each visible side of each block give the window type and an
  estimated window-to-wall ratio.
- List in `assumptions` what the images do not show (back facades,
  depth) and what you assumed for it.
- Write `reading` first, then the other fields consistently with it.
"""


def load_image_part(path: str) -> ImageContentPart:
    """An image file as a base64 content part.

    Raises:
        ValueError: If the file type is not a supported image type.
    """
    p = Path(path)
    mime = IMAGE_TYPES.get(p.suffix.lower())
    if mime is None:
        raise ValueError(
            f"{p.name}: unsupported image type; use {', '.join(IMAGE_TYPES)}"
        )
    return ImageContentPart(
        type="image",
        source_type="base64",
        mime_type=mime,
        data=base64.b64encode(p.read_bytes()).decode("ascii"),
    )


def observe_node(state: AgentState) -> AgentStateUpdate:
    """Read the images with the vision model; nothing to do without images.

    Raises:
        ValueError: If images are given but no vision model is configured,
            before any LLM call.
    """
    if not state.image_paths or state.photo_reading is not None:
        return AgentStateUpdate()
    llm = create_vision_llm()
    parts: list[ContentPart] = [TextContentPart(type="text", text=state.user_input)]
    parts += [load_image_part(path) for path in state.image_paths]
    images = HumanMessage(content=cast("list[str | dict[str, Any]]", parts))
    system = SystemMessage(content=OBSERVE_SYSTEM_PROMPT + language_directive())
    reading, _ = structured(llm, PhotoReadingSchema, [system, images], lambda r: r)
    return AgentStateUpdate(photo_reading=reading)
