from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from src.agent.llm import create_vision_llm
from src.agent.nodes import intake as intake_module
from src.agent.nodes import observe as observe_module
from src.agent.nodes.observe import load_image_part, observe_node
from src.agent.state import AgentState, PhotoReadingSchema
from src.configs.config import LLMConfig

READING = PhotoReadingSchema.model_validate(
    {
        "reading": "3 podium bands, 12 tower bands above",
        "total_storeys": 15,
        "blocks": [
            {"name": "podium", "role": "podium", "storeys": 3, "bottom_storey": 1,
             "width_m": 60, "depth_m": 40, "position": "base"},
            {"name": "tower", "role": "tower", "storeys": 12, "bottom_storey": 4,
             "width_m": 50, "depth_m": 22, "position": "on the podium"},
        ],
        "facades": [{"block": "tower", "side": "front",
                     "windows": "continuous ribbon", "window_to_wall_ratio": 0.45}],
        "assumptions": ["back facade like the front"],
    }
)  # fmt: skip


def _config(vision: str | None) -> LLMConfig:
    return LLMConfig(
        provider="openai",
        model_name="main-model",
        vision_model_name=vision,
        temperature=0.7,
        max_tokens=100,
        api_key="test",
    )


def test_without_images_nothing_is_read_and_no_vision_model_is_needed(monkeypatch):
    def unexpected() -> None:
        raise AssertionError("no vision model without images")

    monkeypatch.setattr(observe_module, "create_vision_llm", unexpected)

    assert observe_node(AgentState(user_input="brief")) == {}


def test_images_without_a_vision_model_are_refused():
    with pytest.raises(ValueError, match="LLM_VISION_MODEL"):
        create_vision_llm(_config(None))


def test_the_vision_model_is_the_configured_one_without_temperature():
    llm: Any = create_vision_llm(_config("vision-model"))

    assert llm.model_name == "vision-model"
    assert llm.temperature is None


def test_images_are_read_once_with_the_vision_model(monkeypatch, tmp_path):
    photo = tmp_path / "front.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    sent: list = []

    def structured(llm, schema, messages, check):
        sent.append((llm, schema, messages))
        return READING, check(READING)

    monkeypatch.setattr(observe_module, "create_vision_llm", lambda: "vision")
    monkeypatch.setattr(observe_module, "structured", structured)

    update = observe_node(AgentState(user_input="brief", image_paths=[str(photo)]))

    assert update == {"photo_reading": READING}
    [(llm, schema, messages)] = sent
    assert (llm, schema) == ("vision", PhotoReadingSchema)
    text, image = messages[-1].content
    assert text == {"type": "text", "text": "brief"}
    assert (image["type"], image["mime_type"]) == ("image", "image/jpeg")


def test_unsupported_image_types_are_refused(tmp_path):
    drawing = tmp_path / "plan.pdf"
    drawing.write_bytes(b"%PDF")

    with pytest.raises(ValueError, match="unsupported image type"):
        load_image_part(str(drawing))


def test_intake_gets_the_reading_as_text_and_no_image(tmp_path):
    state = AgentState(
        user_input="An office in Tokyo.",
        image_paths=[str(tmp_path / "front.jpg")],
        photo_reading=READING,
    )

    brief: HumanMessage = intake_module._brief(state)

    assert isinstance(brief.content, str)
    assert brief.content.startswith("An office in Tokyo.")
    assert '"total_storeys": 15' in brief.content
    assert intake_module._brief(AgentState(user_input="brief")).content == "brief"
