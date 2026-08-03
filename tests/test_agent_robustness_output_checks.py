from tests.agent.test_run_robustness import _check_simulation_output


def test_eplustbl_alone_is_not_successful_simulation_output(tmp_path):
    (tmp_path / "eplustbl.csv").write_text("table only", encoding="utf-8")
    (tmp_path / "eplusout.end").write_text(
        "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors",
        encoding="utf-8",
    )

    ok, artifacts, err_path = _check_simulation_output(tmp_path)

    assert ok is False
    assert any(path.endswith("eplustbl.csv") for path in artifacts)
    assert err_path is None


def test_end_plus_timeseries_is_successful_simulation_output(tmp_path):
    (tmp_path / "eplusout.end").write_text(
        "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors",
        encoding="utf-8",
    )
    (tmp_path / "eplusout.csv").write_text("Date/Time,Zone Temp\n", encoding="utf-8")

    ok, artifacts, err_path = _check_simulation_output(tmp_path)

    assert ok is True
    assert any(path.endswith("eplusout.csv") for path in artifacts)
    assert err_path is None
