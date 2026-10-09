import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.runner.runner import EnergyPlusMessage, run_energyplus

_ARG_PARSER = """#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in -d) out="$2"; shift ;; esac
  shift
done
pwd > "$out/cwd.txt"
"""

_ERR_FILE = """Program Version,EnergyPlus, Version 26.1.0
   ** Warning ** GetVertices: Floor is upside down! Tilt angle=[0.0]
   **   ~~~   ** Automatic fix is attempted.
   ** Severe  ** ProcessScheduleInput: incomplete day detected, Schedule=OCC
   **   ~~~   ** ref Schedule:Compact="OCC"
   ** Severe  ** ProcessScheduleInput: incomplete day detected, Schedule=OCC
   **   ~~~   ** ref Schedule:Compact="OCC"
   **  Fatal  ** ProcessScheduleInput: Preceding Errors cause termination.
   ************* EnergyPlus Terminated--Fatal Error Detected. 1 Warning; 2 Severe Errors
"""


def _fake_energyplus(bin_dir: Path, body: str) -> None:
    script = bin_dir / "energyplus"
    script.write_text(_ARG_PARSER + body, encoding="utf-8")
    script.chmod(0o755)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    idf_path = tmp_path / "in.idf"
    epw_path = tmp_path / "weather.epw"
    idf_path.touch()
    epw_path.touch()
    return idf_path, epw_path


def test_run_parses_err_and_runs_inside_output_dir(tmp_path, monkeypatch, inputs):
    (tmp_path / "bin").mkdir()
    (tmp_path / "expected.err").write_text(_ERR_FILE, encoding="utf-8")
    # 400 KB of stdout must be drained while the process runs.
    _fake_energyplus(
        tmp_path / "bin",
        f'yes "progress line" | head -n 25000\ncp {tmp_path}/expected.err "$out/eplusout.err"\n',
    )
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    output_dir = tmp_path / "run"

    idf_path, epw_path = inputs
    result = run_energyplus(idf_path, epw_path, output_dir)

    assert result.return_code == 0
    assert not result.succeeded
    assert (output_dir / "cwd.txt").read_text().strip() == str(output_dir)
    assert result.errors == (
        EnergyPlusMessage(
            "Severe",
            'ProcessScheduleInput: incomplete day detected, Schedule=OCC\nref Schedule:Compact="OCC"',
            count=2,
        ),
        EnergyPlusMessage(
            "Fatal", "ProcessScheduleInput: Preceding Errors cause termination."
        ),
    )
    assert result.messages[0].text.endswith("Automatic fix is attempted.")


def test_run_kills_process_on_timeout(tmp_path, monkeypatch, inputs):
    (tmp_path / "bin").mkdir()
    _fake_energyplus(tmp_path / "bin", "exec sleep 30\n")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")

    idf_path, epw_path = inputs
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_energyplus(idf_path, epw_path, tmp_path / "run", timeout_s=0.5)
    assert time.perf_counter() - started < 10


def test_run_without_err_file_raises(tmp_path, monkeypatch, inputs):
    (tmp_path / "bin").mkdir()
    _fake_energyplus(tmp_path / "bin", "exit 1\n")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")

    idf_path, epw_path = inputs
    with pytest.raises(FileNotFoundError, match="code 1"):
        run_energyplus(idf_path, epw_path, tmp_path / "run")


@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
def test_parallel_runs_with_expand_objects_succeed(tmp_path):
    # Concurrent runs sharing one working directory abort inside ExpandObjects.
    install = Path(shutil.which("energyplus") or "").resolve().parent
    idf_path = install / "ExampleFiles" / "HVACTemplate-5ZonePurchAir.idf"
    epw_path = next((install / "WeatherData").glob("*.epw"))

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(run_energyplus, idf_path, epw_path, tmp_path / f"run{i}")
            for i in range(3)
        ]
        results = [future.result() for future in futures]

    assert all(result.succeeded for result in results)
