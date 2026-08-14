import json

from scripts import e2e_multiturn


def _revision_properties(*, south_wwr: float, office_lpd: float) -> dict:
    preserved_objects = {
        key: [{"type": key, "name": key, "fields": {"value": 1}}]
        for key in (
            "zones",
            "materials",
            "constructions",
            "thermostats",
            "ideal_loads",
        )
    }
    return {
        "south_wwr": south_wwr,
        "office_lights": {
            "Office_Lights": {
                "zone_name": "OpenOffice",
                "watts_per_floor_area": office_lpd,
            }
        },
        "preserved_objects": preserved_objects,
    }


def test_main_retains_report_and_fails_when_revision_assertion_fails(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "output"
    turn1_idf = tmp_path / "turn1.idf"
    turn2_idf = tmp_path / "turn2.idf"
    turn1_idf.write_text("turn 1", encoding="utf-8")
    turn2_idf.write_text("turn 2", encoding="utf-8")
    turns = iter([({}, turn1_idf), ({}, turn2_idf)])

    monkeypatch.setattr(e2e_multiturn, "OUT", output_dir)
    monkeypatch.setattr(e2e_multiturn, "build_graph", object)
    monkeypatch.setattr(e2e_multiturn, "run_turn", lambda *args, **kwargs: next(turns))
    monkeypatch.setattr(
        e2e_multiturn,
        "_object_inventory",
        lambda path: {"zones": {"count": 1, "sample_names": ["OpenOffice"]}},
    )
    monkeypatch.setattr(
        e2e_multiturn,
        "_revision_properties",
        lambda path: (
            _revision_properties(south_wwr=0.30, office_lpd=10.0)
            if path == turn1_idf
            else _revision_properties(south_wwr=0.25, office_lpd=12.0)
        ),
    )

    exit_code = e2e_multiturn.main()

    report_path = output_dir / "e2e_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["assertions"]["passed"] is False
    assert "turn-2 south WWR" in report["assertions"]["error"]
