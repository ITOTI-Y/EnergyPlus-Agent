"""Run EnergyPlus as a subprocess and collect its eplusout.err diagnostics."""

import shutil
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, Literal, cast

from src.utils.logging import get_logger

logger = get_logger(__name__)

SIMULATION_TIMEOUT_S: Final = 3600.0

type Severity = Literal["Warning", "Severe", "Fatal"]

_SEVERITY_PREFIXES: Final[dict[str, Severity]] = {
    "** Warning **": "Warning",
    "** Severe  **": "Severe",
    "**  Fatal  **": "Fatal",
}
_CONTINUATION_PREFIX: Final = "**   ~~~   **"


@dataclass(frozen=True, slots=True)
class EnergyPlusMessage:
    """One eplusout.err message with its ``~~~`` continuation lines.

    Identical messages are merged; ``count`` records how often they occurred.
    """

    severity: Severity
    text: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class SimulationResult:
    output_dir: Path
    return_code: int
    messages: tuple[EnergyPlusMessage, ...]

    @property
    def errors(self) -> tuple[EnergyPlusMessage, ...]:
        return tuple(m for m in self.messages if m.severity != "Warning")

    @property
    def succeeded(self) -> bool:
        # EnergyPlus can exit 0 while reporting Severe errors that invalidate
        # the results, so both signals are required.
        return self.return_code == 0 and not self.errors


def parse_err_file(err_path: Path) -> tuple[EnergyPlusMessage, ...]:
    """Parse eplusout.err into messages, merging identical repeats."""
    blocks: list[tuple[Severity, list[str]]] = []
    for raw_line in err_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = next(
            (item for item in _SEVERITY_PREFIXES.items() if line.startswith(item[0])),
            None,
        )
        if match is not None:
            prefix, severity = match
            blocks.append((severity, [line.removeprefix(prefix).strip()]))
        elif line.startswith(_CONTINUATION_PREFIX) and blocks:
            blocks[-1][1].append(line.removeprefix(_CONTINUATION_PREFIX).strip())

    counts = Counter((severity, "\n".join(lines)) for severity, lines in blocks)
    return tuple(
        EnergyPlusMessage(severity, text, count)
        for (severity, text), count in counts.items()
    )


def _log_output(stream: IO[str]) -> None:
    for line in stream:
        logger.info("[EnergyPlus] {}", line.rstrip())


def run_energyplus(
    idf_path: Path,
    epw_path: Path,
    output_dir: Path,
    *,
    timeout_s: float = SIMULATION_TIMEOUT_S,
) -> SimulationResult:
    """Run one EnergyPlus simulation with ExpandObjects and ReadVarsESO.

    Args:
        idf_path: Input IDF file.
        epw_path: Weather file.
        output_dir: Directory dedicated to this run; it is also the working
            directory of the process.
        timeout_s: Wall-clock limit before the process is killed.

    Returns:
        Exit code and the parsed eplusout.err messages.

    Raises:
        FileNotFoundError: If EnergyPlus is not on PATH, an input file is
            missing, or the run produced no eplusout.err.
        TimeoutError: If the run exceeds ``timeout_s``.
    """
    executable = shutil.which("energyplus")
    if executable is None:
        raise FileNotFoundError("EnergyPlus executable not found on PATH")
    idf_path = idf_path.resolve(strict=True)
    epw_path = epw_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        executable,
        "-x",
        "-r",
        "-w",
        str(epw_path),
        "-d",
        str(output_dir),
        str(idf_path),
    ]
    logger.info("Running EnergyPlus: {}", " ".join(cmd))
    # ExpandObjects writes intermediate files into the working directory;
    # concurrent runs sharing one cwd abort, so each run uses its own output dir.
    process = subprocess.Popen(
        cmd,
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # stdout=PIPE guarantees a stream; Popen types it as optional.
    stdout = cast(IO[str], process.stdout)
    reader = threading.Thread(target=_log_output, args=(stdout,), daemon=True)
    reader.start()
    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise TimeoutError(
            f"EnergyPlus exceeded {timeout_s:g} s for {idf_path}"
        ) from exc
    finally:
        reader.join()

    err_path = output_dir / "eplusout.err"
    if not err_path.exists():
        raise FileNotFoundError(
            f"EnergyPlus exited with code {return_code} without writing {err_path}"
        )
    result = SimulationResult(output_dir, return_code, parse_err_file(err_path))
    logger.info(
        "EnergyPlus finished: exit code {}, {} error message(s)",
        return_code,
        len(result.errors),
    )
    return result
