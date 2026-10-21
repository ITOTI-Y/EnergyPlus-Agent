import base64
import io
import json
from contextlib import nullcontext

import numpy as np
import pytest
import trimesh
from PIL import Image

from src.agent import mesh_massing
from src.agent.mesh_massing import (
    UnusableMeshError,
    blocks_from_mesh,
    generate_mesh,
    roof_height,
    with_mesh_blocks,
)
from src.agent.state import PhotoReadingSchema


def _box(
    x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]
) -> trimesh.Trimesh:
    """A box from coordinate ranges, glTF axes: y up, front towards +z."""
    size = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
    centre = [(x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2]
    return trimesh.creation.box(
        size, trimesh.transformations.translation_matrix(centre)
    )


def _building() -> trimesh.Trimesh:
    """A 40 x 14 m slab 55.5 m high, a 7-storey podium 16 m deep in front of
    it, and a 10 m antenna on the roof; scaled down, as a 3D model's mesh."""
    slab = _box((0, 40), (0, 55.5), (-14, 0))
    podium = _box((0, 40), (0, 24.9), (0, 16))
    antenna = _box((20, 20.5), (55.5, 65.5), (-7.5, -7))
    mesh = trimesh.Trimesh(
        *trimesh.util.append_faces(
            [m.vertices for m in (slab, podium, antenna)],
            [m.faces for m in (slab, podium, antenna)],
        )
    )
    mesh.apply_scale(0.02)
    return mesh


def _glb(geometry) -> bytes:
    data = geometry.export(file_type="glb")
    assert isinstance(data, bytes)
    return data


READING = PhotoReadingSchema.model_validate(
    {
        "reading": "16 storeys on the end wall",
        "storey_rows": [f"r{k}" for k in range(16, 0, -1)],
        "ground_storey_height_m": 4.5,
        "storey_height_m": 3.4,
        "blocks": [
            {"name": "tower", "role": "tower", "storeys": 16, "bottom_storey": 1, "width_m": 30,
             "depth_m": 20, "x_m": 0, "front_setback_m": 0, "volumes": [1]},
            {"name": "base", "role": "podium", "storeys": 2, "bottom_storey": 1, "width_m": 30,
             "depth_m": 8, "x_m": 0, "front_setback_m": -8, "volumes": [2]},
        ],
        "facades": [
            {"block": "tower", "side": "front", "space": "office", "windows": "ribbon", "pattern": "ribbon", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.5},
            {"block": "base", "side": "front", "space": "office", "windows": "curtain wall", "pattern": "ribbon", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.7},
        ],
        "assumptions": [],
    }
)  # fmt: skip


def test_the_roof_is_where_the_slab_ends_not_the_antenna():
    plan, heights = mesh_massing._surface_points(_building())

    roof = roof_height(*mesh_massing._profile(plan, heights))

    assert roof == pytest.approx(55.5 * 0.02, rel=0.02)


def test_the_slab_and_the_podium_in_front_become_blocks_on_the_storeys():
    slab, podium = blocks_from_mesh(_building(), READING)

    assert (slab.role, slab.bottom_storey, slab.storeys) == ("tower", 1, 16)
    assert (slab.width_m, slab.depth_m) == pytest.approx((40, 14), abs=1.5)
    assert (podium.role, podium.bottom_storey, podium.storeys) == ("podium", 1, 7)
    assert (podium.width_m, podium.depth_m) == pytest.approx((40, 16), abs=1.5)
    # The podium stands in front of the slab, its front 16 m further south.
    assert podium.front_setback_m == pytest.approx(-16, abs=1.5)
    assert podium.x_m == pytest.approx(0, abs=1.5)


def test_mesh_blocks_take_the_facades_of_the_read_blocks_they_match():
    reading = with_mesh_blocks(READING, _building())

    slab, podium = reading.blocks
    assert [(f.block, f.window_to_wall_ratio) for f in reading.facades] == [
        (slab.name, 0.5),
        (podium.name, 0.7),
    ]
    assert len(reading.storey_rows) == 16
    assert "3D model" in reading.assumptions[-1]


def test_a_mesh_whose_top_is_too_small_to_model_is_refused():
    # A 25 m2 spire on a podium: the top storey has nothing to model.
    podium = _box((0, 40), (0, 24.9), (0, 16))
    spire = _box((18, 23), (24.9, 55.5), (6, 11))
    mesh = trimesh.Trimesh(
        *trimesh.util.append_faces(
            [m.vertices for m in (podium, spire)], [m.faces for m in (podium, spire)]
        )
    )

    with pytest.raises(UnusableMeshError, match="top storey"):
        blocks_from_mesh(mesh, READING)


def test_a_filled_box_is_not_taken_as_a_building():
    with pytest.raises(UnusableMeshError, match="filled box"):
        blocks_from_mesh(_box((0, 2), (0, 2), (0, 1.2)), READING)


def _tokenhub(monkeypatch, glb: bytes, final: dict) -> list:
    calls = []
    replies = iter(
        [
            {"id": "job1", "status": "queued"},
            {"id": "job1", "status": "in_progress"},
            {"id": "job1", **final},
        ]
    )

    def urlopen(request, timeout):
        if isinstance(request, str):
            calls.append((request, None, None))
            return nullcontext(io.BytesIO(glb))
        calls.append(
            (
                request.full_url,
                request.get_header("Authorization"),
                json.loads(request.data),
            )
        )
        return nullcontext(io.BytesIO(json.dumps(next(replies)).encode()))

    monkeypatch.setattr(mesh_massing.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(mesh_massing, "POLL_INTERVAL_S", 0)
    return calls


def test_the_image_is_submitted_polled_and_its_mesh_downloaded(monkeypatch):
    glb = _glb(_box((0, 1), (0, 2), (0, 1)))
    done = {
        "status": "completed",
        "data": [{"type": "glb", "url": "https://cos.example/job1_0.glb?q-sign=x"}],
    }
    calls = _tokenhub(monkeypatch, glb, done)

    mesh = generate_mesh(Image.new("RGB", (64, 64)), "https://hub.example/3d/", "k")

    submit, poll, last_poll, download = calls
    assert submit[0] == "https://hub.example/3d/submit"
    assert submit[1] == "Bearer k"
    body = submit[2]
    assert (body["generate_type"], body["face_count"]) == ("Geometry", 20_000)
    image = base64.b64decode(body["image_base64"])
    assert Image.open(io.BytesIO(image)).size == (64, 64)
    assert poll[0] == last_poll[0] == "https://hub.example/3d/query"
    assert poll[2] == {"model": body["model"], "id": "job1"}
    assert download[0] == "https://cos.example/job1_0.glb?q-sign=x"
    assert np.allclose(mesh.extents, (1, 2, 1))


def test_a_failed_job_is_refused(monkeypatch):
    _tokenhub(monkeypatch, b"", {"status": "failed"})

    with pytest.raises(ValueError, match="gave no mesh"):
        generate_mesh(Image.new("RGB", (64, 64)), "https://hub.example/3d", "k")


def test_a_reply_without_a_mesh_is_refused(monkeypatch):
    points = _glb(trimesh.PointCloud(np.random.default_rng(0).random((10, 3))))
    done = {"status": "completed", "data": [{"type": "glb", "url": "https://c/x.glb"}]}
    _tokenhub(monkeypatch, points, done)

    with pytest.raises(ValueError, match="no mesh"):
        generate_mesh(Image.new("RGB", (64, 64)), "https://hub.example/3d", "k")


def test_the_main_mesh_block_takes_the_slab_facades_even_when_a_core_is_read_higher():
    # Read: a ribbon-windowed slab on storeys 3-14 with a taller, almost
    # windowless core at its east end, and a low base in front.
    reading = PhotoReadingSchema.model_validate(
        READING.model_dump()
        | {
            "blocks": [
                {"name": "slab", "role": "tower", "storeys": 12, "bottom_storey": 3, "width_m": 35,
                 "depth_m": 20, "x_m": 0, "front_setback_m": 0, "volumes": []},
                {"name": "core", "role": "core", "storeys": 16, "bottom_storey": 1, "width_m": 12,
                 "depth_m": 20, "x_m": 35, "front_setback_m": 0, "volumes": []},
                {"name": "base", "role": "podium", "storeys": 2, "bottom_storey": 1, "width_m": 30,
                 "depth_m": 8, "x_m": 0, "front_setback_m": -8, "volumes": []},
            ],
            "facades": [
                {"block": "slab", "side": "front", "space": "office", "windows": "ribbon", "pattern": "ribbon", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.5},
                {"block": "core", "side": "front", "space": "stair or service core", "windows": "punched", "pattern": "punched", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.05},
                {"block": "base", "side": "front", "space": "office", "windows": "curtain wall", "pattern": "ribbon", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.4},
            ],
        }
    )  # fmt: skip

    meshed = with_mesh_blocks(reading, _building())

    slab, podium = meshed.blocks
    assert {(f.block, f.window_to_wall_ratio) for f in meshed.facades} == {
        (slab.name, 0.5),
        (podium.name, 0.4),
    }


def _without_horizontal_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Walls only, as an open shell: no roof, floor or other flat face."""
    walls = np.abs(mesh.face_normals[:, 1]) < 0.5
    return trimesh.Trimesh(mesh.vertices, mesh.faces[walls])


def test_an_open_shell_still_gives_solid_storey_plans():
    slab, podium = blocks_from_mesh(_without_horizontal_faces(_building()), READING)

    assert (slab.width_m, slab.depth_m) == pytest.approx((40, 14), abs=2)
    assert (podium.bottom_storey, podium.storeys) == (1, 7)


def test_a_building_seen_from_its_corner_is_turned_to_the_read_proportions():
    corner = _building()
    corner.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
    )

    slab, _ = blocks_from_mesh(corner, READING)  # the read main block is wide

    assert (slab.width_m, slab.depth_m) == pytest.approx((40, 14), abs=2.5)


def test_the_end_of_a_mesh_block_takes_the_facade_of_the_read_block_ending_there():
    # Read: a ribbon slab whose east end is a core with a column of windows.
    reading = PhotoReadingSchema.model_validate(
        READING.model_dump()
        | {
            "blocks": [
                {"name": "slab", "role": "tower", "storeys": 16, "bottom_storey": 1, "width_m": 30,
                 "depth_m": 14, "x_m": 0, "front_setback_m": 0, "volumes": []},
                {"name": "core", "role": "core", "storeys": 16, "bottom_storey": 1, "width_m": 10,
                 "depth_m": 14, "x_m": 30, "front_setback_m": 0, "volumes": []},
            ],
            "facades": [
                {"block": "slab", "side": "front", "space": "office", "windows": "ribbon", "pattern": "ribbon", "window_width_m": None, "window_height_m": None,
                 "window_to_wall_ratio": 0.5},
                {"block": "core", "side": "right", "space": "stair or service core",
                 "windows": "punched", "pattern": "punched", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.05},
            ],
        }
    )  # fmt: skip

    meshed = with_mesh_blocks(reading, _building())

    slab = meshed.blocks[0]
    sides = {
        (f.side, f.window_to_wall_ratio) for f in meshed.facades if f.block == slab.name
    }
    assert sides == {("front", 0.5), ("right", 0.05)}


def test_a_podium_ending_with_the_tower_leaves_the_end_to_the_core():
    reading = PhotoReadingSchema.model_validate(
        READING.model_dump()
        | {
            "blocks": [
                {"name": "slab", "role": "tower", "storeys": 16, "bottom_storey": 1, "width_m": 30,
                 "depth_m": 14, "x_m": 0, "front_setback_m": 0, "volumes": []},
                {"name": "core", "role": "core", "storeys": 16, "bottom_storey": 1, "width_m": 8,
                 "depth_m": 14, "x_m": 30, "front_setback_m": 0, "volumes": []},
                {"name": "podium", "role": "podium", "storeys": 2, "bottom_storey": 1, "width_m": 40,
                 "depth_m": 20, "x_m": 0, "front_setback_m": -6, "volumes": []},
            ],
            "facades": [
                {"block": "slab", "side": "front", "space": "office", "windows": "ribbon", "pattern": "ribbon", "window_width_m": None, "window_height_m": None,
                 "window_to_wall_ratio": 0.5},
                {"block": "core", "side": "right", "space": "stair or service core",
                 "windows": "punched", "pattern": "punched", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.05},
                {"block": "podium", "side": "right", "space": "lobby or retail",
                 "windows": "storey-high glazing", "pattern": "ribbon", "window_width_m": None, "window_height_m": None, "window_to_wall_ratio": 0.6},
            ],
        }
    )  # fmt: skip

    meshed = with_mesh_blocks(reading, _building())

    slab = meshed.blocks[0]
    east = [f for f in meshed.facades if f.block == slab.name and f.side == "right"]
    assert [f.window_to_wall_ratio for f in east] == [0.05]
