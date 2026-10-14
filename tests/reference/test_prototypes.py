import zipfile

import pytest

from src.reference import prototypes

HEADER = "! Case = ASHRAE901_OfficeSmall_STD2022_HoChiMinh\n!    AnalysisClimateZone = 0A\n!    CZ_Label = 1A\n"
BODY = """
Version,22.1;
Material, Brick, Rough, 0.1, 0.9, 1900, 800;  ! a comment
Construction, Wall, Brick;
BuildingSurface:Detailed, W1, Wall, Wall, Zone 1, , Outdoors, , SunExposed, WindExposed, autocalculate, 4;
FenestrationSurface:Detailed, Win1, Window, Glass, W1;
InternalMass, Furniture 1, Furnishings, Zone 1, 40;
Zone, Zone 1;
"""


def test_reduced_model_keeps_library_objects_at_the_target_version():
    text = prototypes.upgrade(prototypes.reduce_model(HEADER + BODY))

    assert "Version,26.1;" in text.replace(" ", "")
    assert "Material" in text and "Construction" in text
    assert "Zone" not in text and "BuildingSurface" not in text
    assert "comment" not in text


def test_models_older_than_22_1_are_refused():
    with pytest.raises(ValueError, match="predates"):
        prototypes.upgrade("Version,8.0;")


def test_construction_uses_come_from_the_original_surfaces():
    roles = prototypes.construction_roles(HEADER + BODY)

    assert roles == {
        "wall": {"Wall, Outdoors"},
        "glass": {"Window"},
        "furnishings": {"InternalMass"},
    }


def test_archives_are_labelled_by_file_name_and_header(tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("x/ASHRAE901_OfficeSmall_STD2022_HoChiMinh.idf", HEADER + BODY)
        bundle.writestr("US+SF+CZ4AWH+hp+slab+IECC_2021.idf", BODY)
        bundle.writestr("US+SF+CZ4AWHT+hp+slab+IECC_2021.idf", BODY)  # same label
        bundle.writestr("US+SF+CZ4AWH+gasfurnace+slab+IECC_2021.idf", BODY)
        bundle.writestr("US+SF+CZ4AWH+hp+slab+IECC_2018.idf", BODY)

    models = prototypes.extract([archive], tmp_path / "out")

    assert [
        (m.category, m.building_type, m.standard, m.climate_zone, m.location)
        for m in models
    ] == [
        ("residential", "SingleFamily_slab", "IECC 2021", "4A", "CZ4A"),
        ("commercial", "OfficeSmall", "ASHRAE 90.1-2022", "0A", "HoChiMinh"),
    ]
    assert models[1].roles["glass"] == {"Window"}
