from idfpy import IDF

from src.runner.runner import EnergyPlusRunner


def test_run_idf_accepts_in_memory_idf(tmp_path, monkeypatch):
    energyplus = tmp_path / "energyplus"
    energyplus.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    energyplus.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    epw_path = tmp_path / "weather.epw"
    epw_path.touch()

    runner = EnergyPlusRunner(IDF())

    assert runner.run_idf(epw_path, output_directory=tmp_path / "output") is True
