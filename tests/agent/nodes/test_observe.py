from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from src.agent.llm import create_vision_llm
from src.agent.nodes import intake as intake_module
from src.agent.nodes import observe as observe_module
from src.agent.nodes.intake import with_reading_storeys
from src.agent.nodes.observe import load_image_part, observe_node
from src.agent.state import AgentState, PhotoReadingSchema, ZonePlanSchema
from src.agent.storeys import storeys_from_reading
from src.configs.config import LLMConfig
from tests.agent.intake_data import intake, zone_spec

READING = PhotoReadingSchema.model_validate(
    {
        "reading": "3 podium bands, 12 tower bands above",
        "total_storeys": 15,
        "ground_storey_height_m": 4.5,
        "storey_height_m": 3.8,
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


def _plan(key: str, block: str | None, corners: list) -> ZonePlanSchema:
    plan = zone_spec(key, corners) | {"key": key, "block": block}
    return ZonePlanSchema.model_validate(
        {k: v for k, v in plan.items() if k not in ("name", "floor_z", "height")}
    )


PODIUM = _plan("Podium", "podium", [(0, 0), (60, 0), (60, 40), (0, 40)])
TOWER = _plan("Tower", "tower", [(0, 9), (50, 9), (50, 31), (0, 31)])


def test_storeys_follow_the_blocks_as_ground_typical_and_top():
    storeys = storeys_from_reading([PODIUM, TOWER], READING)

    # Podium storeys 1-3 (ground 4.5 m), tower 4-15 (3.8 m).
    assert [
        (s.name, s.height, s.multiplier, [z.plan for z in s.zones]) for s in storeys
    ] == [
        ("S1", 4.5, 1, ["Podium"]),
        ("S2", 3.8, 1, ["Podium"]),
        ("S3", 3.8, 1, ["Podium"]),
        ("S4", 3.8, 1, ["Tower"]),
        ("S5", 3.8, 10, ["Tower"]),
        ("S15", 3.8, 1, ["Tower"]),
    ]
    assert sum(s.multiplier for s in storeys) == 15


def test_a_core_beside_the_whole_height_shares_every_storey():
    core = _plan("Core", "core", [(50, 9), (62, 9), (62, 31), (50, 31)])
    reading = READING.model_copy(
        update={
            "blocks": [
                *READING.blocks,
                READING.blocks[0].model_copy(
                    update={"name": "core", "role": "core", "storeys": 15}
                ),
            ]
        }
    )

    storeys = storeys_from_reading([PODIUM, TOWER, core], reading)

    assert [[z.plan for z in s.zones] for s in storeys][2:4] == [
        ["Podium", "Core"],
        ["Tower", "Core"],
    ]


def test_plans_and_blocks_that_do_not_match_go_back_to_intake():
    with pytest.raises(ValueError, match="sets `block`"):
        storeys_from_reading([PODIUM, _plan("Tower", None, TOWER_CORNERS)], READING)
    with pytest.raises(ValueError, match="without a plan: \\['tower'\\]"):
        storeys_from_reading([PODIUM], READING)


TOWER_CORNERS = [(0, 9), (50, 9), (50, 31), (0, 31)]


def test_a_gap_between_blocks_takes_the_plans_below():
    gap = READING.model_copy(
        update={
            "blocks": [
                READING.blocks[0],
                READING.blocks[1].model_copy(update={"bottom_storey": 6}),
            ]
        }
    )

    storeys = storeys_from_reading([PODIUM, TOWER], gap)

    # Podium 1-3, nothing on 4-5, tower 6-17: 4-5 extend the podium, whose
    # storeys 2-5 are then one run (the ground storey is higher).
    assert [(s.name, s.multiplier, s.zones[0].plan) for s in storeys][:4] == [
        ("S1", 1, "Podium"),
        ("S2", 1, "Podium"),
        ("S3", 2, "Podium"),
        ("S5", 1, "Podium"),
    ]
    assert sum(s.multiplier for s in storeys) == 17


def test_intake_output_gets_the_storeys_built_from_the_reading():
    without = intake(
        zone_plans=[PODIUM.model_dump(by_alias=True), TOWER.model_dump(by_alias=True)],
        storeys=[],
    )

    built = with_reading_storeys(without, READING)

    assert len(built.zones) == 6
    assert built.zones[0].name == "S1_Podium"
    assert with_reading_storeys(without, None) is without
