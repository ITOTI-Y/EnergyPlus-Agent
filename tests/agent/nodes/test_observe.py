import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from langchain_core.messages import HumanMessage
from PIL import Image

from src.agent.llm import create_vision_llm
from src.agent.nodes import intake as intake_module
from src.agent.nodes import observe as observe_module
from src.agent.nodes.intake import with_reading_storeys
from src.agent.nodes.observe import (
    ImageContentPart,
    StoreyStripSchema,
    StripCountSchema,
    VolumeSchema,
    VolumesSchema,
    load_image_part,
    observe_node,
)
from src.agent.state import AgentState, IntakePatch, PhotoReadingSchema, ZonePlanSchema
from src.agent.storeys import storeys_from_reading
from src.configs.config import LLMConfig
from tests.agent.intake_data import intake, zone_spec

READING = PhotoReadingSchema.model_validate(
    {
        "reading": "3 podium bands, 12 tower bands above",
        "storey_rows": [f"band {k}" for k in range(15, 0, -1)],
        "ground_storey_height_m": 4.5,
        "storey_height_m": 3.8,
        "blocks": [
            {"name": "podium", "role": "podium", "storeys": 3, "bottom_storey": 1,
             "width_m": 60, "depth_m": 40, "x_m": 0, "front_setback_m": -9, "volumes": [2]},
            {"name": "tower", "role": "tower", "storeys": 12, "bottom_storey": 4,
             "width_m": 50, "depth_m": 22, "x_m": 0, "front_setback_m": 0, "volumes": [1]},
        ],
        "facades": [{"block": "tower", "side": "front",
                     "space": "office", "windows": "continuous ribbon", "pattern": "ribbon", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.45}],
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


def _photo(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, "grey").save(path)
    return path


def _sent_size(part: ImageContentPart) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(part["data"]))).size


def test_small_images_are_enlarged_and_large_ones_keep_their_size(tmp_path):
    small = load_image_part(str(_photo(tmp_path / "small.png", (400, 850))))
    large = load_image_part(str(_photo(tmp_path / "large.png", (2000, 1000))))

    assert (_sent_size(small), small["mime_type"]) == ((800, 1700), "image/jpeg")
    assert _sent_size(large) == (2000, 1000)


def _fake_vision(monkeypatch) -> list:
    """Replies to every vision call; returns the calls as they are made."""
    replies = {
        # Located below the roof; the tower's box reaches it.
        StoreyStripSchema: StoreyStripSchema(bbox_2d=[500, 300, 600, 1000]),
        StripCountSchema: StripCountSchema(storeys=["row"] * 15),
        VolumesSchema: VolumesSchema(
            volumes=[
                VolumeSchema(label="tower", bbox_2d=[400, 0, 700, 800]),
                VolumeSchema(label="base", bbox_2d=[200, 700, 900, 1000]),
            ]
        ),
    }
    sent: list = []

    def structured(llm, schema, messages, check):
        sent.append((llm, schema, messages))
        reply = replies.get(schema, READING)
        return reply, check(reply)

    monkeypatch.setattr(observe_module, "create_vision_llm", lambda: "vision")
    monkeypatch.setattr(observe_module, "structured", structured)
    return sent


def test_the_reading_gets_the_image_its_counted_strip_and_boxed_volumes(
    monkeypatch, tmp_path
):
    photo = _photo(tmp_path / "front.jpg", (1700, 850))
    sent = _fake_vision(monkeypatch)

    update = observe_node(AgentState(user_input="brief", image_paths=[str(photo)]))

    assert update == {"photo_reading": READING}
    assert [s for _, s, _ in sent] == [
        VolumesSchema,
        StoreyStripSchema,
        *[StripCountSchema] * 3,
        PhotoReadingSchema,
    ]
    assert {llm for llm, _, _ in sent} == {"vision"}
    _, counted = sent[2][2][0].content
    # The strip, run up to the roof and a tenth of the image wide, is
    # enlarged to 1700 px high.
    assert _sent_size(counted) == (340, 1700)
    brief, image, note, enlarged, listed, boxed = sent[-1][2][-1].content
    assert "boxed and numbered in the image below: 1 tower; 2 base." in listed["text"]
    assert _sent_size(boxed) == (1700, 850)
    assert boxed["data"] != image["data"]
    assert brief == {"type": "text", "text": "brief"}
    assert _sent_size(image) == (1700, 850)
    assert note["text"].startswith("The facade strip [500, 0, 600, 1000] (0-1000)")
    assert (
        "whole strip from its top to its bottom: [15, 15, 15] storeys, median 15"
        in note["text"]
    )
    assert _sent_size(enlarged) == (340, 1700)


def test_with_an_image_to_3d_server_the_blocks_come_from_its_mesh(
    monkeypatch, tmp_path
):
    photo = _photo(tmp_path / "front.jpg", (1700, 850))
    _fake_vision(monkeypatch)
    meshed = READING.model_copy(update={"assumptions": ["from a mesh"]})
    calls: list = []

    def generate_mesh(image, url, api_key):
        calls.append((image.size, url))
        return "mesh"

    def with_mesh_blocks(reading, mesh):
        calls.append((reading, mesh))
        return meshed

    monkeypatch.setattr(observe_module, "IMAGE_TO_3D_URL", "http://gpu:8081")
    monkeypatch.setattr(observe_module.mesh_massing, "generate_mesh", generate_mesh)
    monkeypatch.setattr(
        observe_module.mesh_massing, "with_mesh_blocks", with_mesh_blocks
    )

    update = observe_node(AgentState(user_input="brief", image_paths=[str(photo)]))

    assert update == {"photo_reading": meshed}
    # Cropped to the volumes, [200, 0, 900, 1000] with a 3% margin.
    assert calls == [((1292, 850), "http://gpu:8081"), (READING, "mesh")]


def test_with_a_segmentation_server_the_3d_model_sees_the_building_cut_out(
    monkeypatch, tmp_path
):
    photo = _photo(tmp_path / "front.jpg", (1700, 850))
    _fake_vision(monkeypatch)
    # Pixels: the target inside the volumes, a neighbour at the right edge.
    target = np.zeros((850, 1700), bool)
    target[100:801, 700:1101] = True
    neighbour = np.zeros((850, 1700), bool)
    neighbour[200:, 1550:] = True
    shown: list = []

    def generate_mesh(image, url, api_key):
        shown.append((image.mode, image.size))
        return "mesh"

    monkeypatch.setattr(observe_module, "IMAGE_TO_3D_URL", "http://gpu:8081")
    monkeypatch.setattr(observe_module, "SEGMENT_URL", "http://gpu:8082")
    monkeypatch.setattr(
        observe_module.segmentation,
        "segment",
        lambda image, prompt, url: [target, neighbour],
    )
    monkeypatch.setattr(observe_module.mesh_massing, "generate_mesh", generate_mesh)
    monkeypatch.setattr(
        observe_module.mesh_massing, "with_mesh_blocks", lambda reading, mesh: reading
    )

    observe_node(AgentState(user_input="brief", image_paths=[str(photo)]))

    assert shown == [("RGBA", (401, 701))]


def test_no_building_in_the_volumes_keeps_the_read_blocks(monkeypatch, tmp_path):
    photo = _photo(tmp_path / "front.jpg", (1700, 850))
    _fake_vision(monkeypatch)

    def unexpected(*_):
        raise AssertionError("no 3D model without a building to show it")

    monkeypatch.setattr(observe_module, "IMAGE_TO_3D_URL", "http://gpu:8081")
    monkeypatch.setattr(observe_module, "SEGMENT_URL", "http://gpu:8082")
    monkeypatch.setattr(
        observe_module.segmentation, "segment", lambda image, prompt, url: []
    )
    monkeypatch.setattr(observe_module.mesh_massing, "generate_mesh", unexpected)

    update = observe_node(AgentState(user_input="brief", image_paths=[str(photo)]))

    reading = update["photo_reading"]
    assert reading is not None
    assert reading.blocks == READING.blocks
    assert reading.assumptions[-1].startswith("No 3D model blocks: no building")


def test_the_3d_model_sees_only_the_volumes_the_blocks_are_made_of(
    monkeypatch, tmp_path
):
    photo = _photo(tmp_path / "front.jpg", (1700, 850))
    _fake_vision(monkeypatch)
    # Volume 2, the wide base, is a neighbour: no block is made of it.
    reading = READING.model_copy(
        update={
            "blocks": [
                b.model_copy(update={"volumes": [1] if b.name == "tower" else []})
                for b in READING.blocks
            ]
        }
    )
    sizes: list = []

    def structured(llm, schema, messages, check):
        reply = {
            StoreyStripSchema: StoreyStripSchema(bbox_2d=[500, 0, 600, 1000]),
            StripCountSchema: StripCountSchema(storeys=["row"] * 15),
            VolumesSchema: VolumesSchema(
                volumes=[
                    VolumeSchema(label="tower", bbox_2d=[400, 0, 700, 800]),
                    VolumeSchema(label="neighbour", bbox_2d=[0, 500, 1000, 1000]),
                ]
            ),
        }.get(schema, reading)
        return reply, check(reply)

    def generate_mesh(image, url, api_key):
        sizes.append(image.size)
        return "mesh"

    def unusable(reading, mesh):
        raise observe_module.mesh_massing.UnusableMeshError("a filled box")

    monkeypatch.setattr(observe_module, "structured", structured)
    monkeypatch.setattr(observe_module, "IMAGE_TO_3D_URL", "http://gpu:8081")
    monkeypatch.setattr(observe_module.mesh_massing, "generate_mesh", generate_mesh)
    monkeypatch.setattr(observe_module.mesh_massing, "with_mesh_blocks", unusable)

    update = observe_node(AgentState(user_input="brief", image_paths=[str(photo)]))

    # The tower's box [400, 0, 700, 800] with a 3% margin, not the neighbour.
    assert sizes == [(612, 705)]
    # An unusable 3D model leaves the reading's blocks and says so.
    kept = update["photo_reading"]
    assert kept is not None and kept.blocks == reading.blocks
    assert kept.assumptions[-1] == "No 3D model blocks: a filled box."


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
    assert '"band 15"' in brief.content
    assert "- tower: x0 0.0, y0 9.0, x1 50.0, y1 31.0" in brief.content
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
    core = _plan("Core", "core", [(60, 9), (72, 9), (72, 31), (60, 31)])
    reading = PhotoReadingSchema.model_validate(
        READING.model_dump()
        | {
            "blocks": [
                *READING.model_dump()["blocks"],
                {"name": "core", "role": "core", "storeys": 15, "bottom_storey": 1,
                 "width_m": 12, "depth_m": 22, "x_m": 60, "front_setback_m": 0, "volumes": []},
            ]
        }
    )  # fmt: skip

    storeys = storeys_from_reading([PODIUM, TOWER, core], reading)

    assert [[z.plan for z in s.zones] for s in storeys][2:4] == [
        ["Podium", "Core"],
        ["Tower", "Core"],
    ]


def test_block_names_match_whatever_dashes_and_spaces_they_are_written_with():
    reading = _with_blocks({"name": "podium\u2011base"}, {"name": "main  tower"})
    podium = PODIUM.model_copy(update={"block": "podium-base"})
    tower = TOWER.model_copy(update={"block": "main tower "})

    assert [b.name for b in reading.blocks] == ["podium base", "main tower"]
    assert storeys_from_reading(
        [ZonePlanSchema.model_validate(p.model_dump()) for p in (podium, tower)],
        reading,
    )


def test_plans_and_blocks_that_do_not_match_go_back_to_intake():
    with pytest.raises(ValueError, match="sets `block`"):
        storeys_from_reading([PODIUM, _plan("Tower", None, TOWER_CORNERS)], READING)
    with pytest.raises(ValueError, match="without a plan: \\['tower'\\]"):
        storeys_from_reading([PODIUM], READING)


def test_plans_off_their_block_footprint_go_back_to_intake():
    flush = _plan("Tower", "tower", [(0, 0), (50, 0), (50, 22), (0, 22)])
    with pytest.raises(ValueError, match=r"'tower' stands on x 0\.0\.\.50\.0, y 9"):
        storeys_from_reading([PODIUM, flush], READING)
    half = _plan("Tower", "tower", [(0, 9), (25, 9), (25, 31), (0, 31)])
    with pytest.raises(ValueError, match="cover 50%"):
        storeys_from_reading([PODIUM, half], READING)


def test_a_plan_crossing_itself_goes_back_to_intake():
    bow = _plan("Tower", "tower", [(0, 9), (50, 31), (50, 9), (0, 31)])
    box_plan = _plan("Tower2", "tower", [(0, 9), (50, 9), (50, 31), (0, 31)])
    with pytest.raises(ValueError, match=r"\['Tower'\] of block 'tower' cross"):
        storeys_from_reading([PODIUM, bow, box_plan], READING)


def _with_blocks(*changes: dict) -> PhotoReadingSchema:
    blocks = [
        b | c for b, c in zip(READING.model_dump()["blocks"], changes, strict=True)
    ]
    return PhotoReadingSchema.model_validate(READING.model_dump() | {"blocks": blocks})


def test_a_lower_block_is_trimmed_beside_a_taller_one_rising_out_of_it():
    # The tower rises out of the podium from storey 3; the podium keeps the
    # strip in front of it.
    reading = _with_blocks({"depth_m": 30}, {"bottom_storey": 3, "storeys": 13})

    assert reading.footprints() == {
        "podium": (0.0, 0.0, 60.0, 9.0),
        "tower": (0.0, 9.0, 50.0, 31.0),
    }


@pytest.mark.parametrize(
    "podium",
    [
        {"width_m": 10, "depth_m": 10, "x_m": 5, "front_setback_m": 5},
        # Only 1 m strips would be left beside the tower.
        {"width_m": 52, "depth_m": 22, "x_m": -1, "front_setback_m": 0},
    ],
    ids=["inside", "slivers"],
)
def test_a_block_inside_a_taller_one_is_dropped(podium):
    reading = _with_blocks(podium, {"bottom_storey": 1, "storeys": 15})

    assert [b.name for b in reading.blocks] == ["tower"]
    assert list(reading.footprints()) == ["tower"]


def test_footprints_place_setbacks_from_the_blocks():
    # The podium projects 9 m in front of the tower and 10 m east of it.
    assert READING.footprints() == {
        "podium": (0.0, 0.0, 60.0, 40.0),
        "tower": (0.0, 9.0, 50.0, 31.0),
    }


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


def test_intake_node_keeps_the_storeys_built_from_the_reading(monkeypatch):
    reply = intake(
        zone_plans=[PODIUM.model_dump(by_alias=True), TOWER.model_dump(by_alias=True)],
        storeys=[],
    )

    def structured(llm, schema, messages, check):
        return reply, check(reply)

    monkeypatch.setattr(intake_module, "create_llm", lambda: None)
    monkeypatch.setattr(intake_module, "structured", structured)
    runtime: Any = SimpleNamespace(context=SimpleNamespace(reference=None))

    update = intake_module.intake_node(
        AgentState(user_input="brief", photo_reading=READING), runtime
    )

    # The reply had no storeys; the node's output has those built in code.
    output = update["intake_output"]
    assert output is not None
    assert [s.name for s in output.storeys] == ["S1", "S2", "S3", "S4", "S5", "S15"]


@pytest.mark.parametrize(
    "plans",
    [
        [PODIUM, TOWER.model_copy(update={"key": "Mast"})],
        [
            PODIUM,
            _plan("Tower", "tower", [(0, 9), (40, 9), (40, 31), (0, 31)]),
            _plan("Annex", "tower", [(40, 9), (50, 9), (50, 31), (40, 31)]),
        ],
    ],
    ids=["renamed", "added"],
)
def test_a_revision_with_a_reading_rebuilds_storeys_and_zones(monkeypatch, plans):
    previous = with_reading_storeys(
        intake(
            zone_plans=[
                PODIUM.model_dump(by_alias=True),
                TOWER.model_dump(by_alias=True),
            ]
        ),
        READING,
    )
    patch = IntakePatch(reason="reviewer", zone_plans=plans)

    def structured(llm, schema, messages, check):
        assert schema is IntakePatch
        return patch, check(patch)

    monkeypatch.setattr(intake_module, "create_llm", lambda: None)
    monkeypatch.setattr(intake_module, "structured", structured)
    runtime: Any = SimpleNamespace(context=SimpleNamespace(reference=None))

    update = intake_module.intake_node(
        AgentState(
            user_input="brief",
            photo_reading=READING,
            intake_output=previous,
            review_feedback="change the tower",
        ),
        runtime,
    )

    output = update["intake_output"]
    assert output is not None
    assert {z.name for z in output.zones} != {z.name for z in previous.zones}
    assert "zone" in update["pending_phases"]
