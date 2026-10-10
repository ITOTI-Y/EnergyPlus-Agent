from idfpy import IDF
from idfpy.models.constructions import Construction, Material

from src.reference import library
from src.reference.prototypes import PrototypeModel


def _model(tmp_path, name, building, zone, conductivity):
    idf = IDF()
    idf.add(
        Material(
            name="Brick",
            roughness="Rough",
            thickness=0.1,
            conductivity=conductivity,
            density=1900.0,
            specific_heat=800.0,
        )
    )
    idf.add(Construction(name="Wall", outside_layer="Brick"))
    path = tmp_path / f"{name}.idf"
    idf.save(path)
    roles = {"wall": {"Wall, Outdoors"}}
    return PrototypeModel(
        "commercial", building, "ASHRAE 90.1-2022", zone, name, path, roles
    )


def test_equal_objects_merge_and_differing_ones_stay_apart(tmp_path):
    models = [
        _model(tmp_path, "a", "OfficeSmall", "1A", 0.9),
        _model(tmp_path, "b", "Warehouse", "2A", 0.9),
        _model(tmp_path, "c", "OfficeSmall", "8", 0.5),
    ]
    path = tmp_path / "library.sqlite"

    report = library.build(models, path, {"source": "test"})

    assert report.entries == {"material": 2, "construction": 2}
    walls = sorted(
        (e for e in library.entries(path) if e.kind == "construction"),
        key=lambda e: e.climate_zones,
    )
    assert [w.climate_zones for w in walls] == [["1A", "2A"], ["8"]]
    # Zone 8 is not everywhere the source models are, so it is named.
    assert "1A very hot" in walls[0].description
    assert "all commercial building types" in walls[0].description
    assert "U-value without surface films 9.00" in walls[0].description


def test_embeddings_round_trip(tmp_path):
    path = tmp_path / "library.sqlite"
    library.build([_model(tmp_path, "a", "OfficeSmall", "1A", 0.9)], path, {})
    key = next(library.entries(path)).id

    library.store_embeddings(path, {key: [0.25, -1.0]})

    assert dict(library.embeddings(path)) == {key: [0.25, -1.0]}
