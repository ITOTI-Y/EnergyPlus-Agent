import contextlib
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from idfpy import IDF

from src.utils.logging import get_logger

# Wall-clock cap on a single EnergyPlus run. EnergyPlus normally finishes a
# year-long run in seconds-to-minutes, but a diverging/ill-conditioned IDF
# can hang the solver indefinitely. Without a timeout the process.wait()
# blocks forever, freezing the whole simulate->revise retry loop (up to 11
# rounds) and leaving a zombie process. 1800s (30 min) is generous enough
# for large buildings yet still bounds the worst-case hang. Override per
# call via run_idf(timeout=...).
DEFAULT_SIMULATION_TIMEOUT_S: int = 1800


class EnergyPlusRunner:
    def __init__(self, idf: IDF | None = None, idd_file_path: Path | None = None):
        """
        Initialize the EnergyPlusRunner.

        Args:
            idf: An instance of idfpy.IDF
            idd_file_path: Unused; kept for backwards-compatible call signatures.
        """
        self.logger = get_logger(__name__)
        self.idf_path: Path | None = None
        self.idf = idf if idf else IDF()
        self.logger.info("EnergyPlusRunner initialized.")

    def run_idf(
        self,
        epw_file_path: Path | str,
        idf_file_path: Path | str | None = None,
        output_directory: Path | None = None,
        timeout: int | None = DEFAULT_SIMULATION_TIMEOUT_S,
    ) -> bool:
        """
        Run EnergyPlus IDF file

        Args:
            idf_file_path: IDF file path
            epw_file_path: EPW weather file path
            output_directory: Output directory, if None, a default directory will be created
            timeout: Max wall-clock seconds to wait for EnergyPlus before
                killing it. Defaults to DEFAULT_SIMULATION_TIMEOUT_S. Pass
                None to disable (not recommended — a hung solver would
                block forever).

        Returns:
            bool: True if the simulation ran successfully, False otherwise
        """
        if idf_file_path:
            self.idf_path = Path(idf_file_path)
            self.idf = IDF.load(self.idf_path)
        elif self.idf_path:
            idf_file_path = self.idf_path
        else:
            raise ValueError(
                "IDF file path must be provided either via parameter or IDF instance."
            )
        self.epw_path = Path(epw_file_path)

        if not self.idf_path.exists():
            raise FileNotFoundError(f"IDF file not found: {self.idf_path}")
        if not self.epw_path.exists():
            raise FileNotFoundError(f"EPW file not found: {self.epw_path}")

        if output_directory is None:
            output_directory = (
                Path(__file__).parent.parent.parent
                / "output"
                / "results"
                / f"energyplus_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting EnergyPlus simulation...")
        self.logger.info("IDF file: {}", self.idf_path)
        self.logger.info("EPW file: {}", self.epw_path)
        self.logger.info("Output directory: {}", output_directory)

        try:
            energyplus_exe = shutil.which("energyplus")
            if not energyplus_exe:
                raise FileNotFoundError("EnergyPlus executable not found in PATH")

            cmd = [
                energyplus_exe,
                "-x",
                "-w",
                str(self.epw_path),
                "-d",
                str(output_directory),
                "-r",
                str(self.idf_path),
            ]

            self.logger.info("Running command: {}", " ".join(cmd))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                output_lines = []

                # Block until the process exits, bounded by `timeout` so a
                # hung solver can't freeze the retry loop forever. Raises
                # TimeoutExpired (caught below) if it overruns.
                if timeout is not None:
                    process.wait(timeout=timeout)
                else:
                    process.wait()

                # Process has exited — drain all buffered stdout (the pipe
                # yields until EOF, reached now that the child is gone).
                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.rstrip()
                        self.logger.info("[EnergyPlus] {}", line)
                        output_lines.append(line)

                return_code = process.returncode

                if return_code != 0:
                    self.logger.error("EnergyPlus exited with code {}", return_code)
                    return False

                self.logger.info("EnergyPlus simulation completed successfully.")
                return True

            except subprocess.TimeoutExpired:
                self.logger.error(
                    "EnergyPlus simulation timed out after {}s; terminating process.",
                    timeout,
                )
                self._terminate(process)
                return False
            finally:
                # Make sure no child is left running and pipes are closed on
                # any exit path (success, non-zero exit, timeout, exception).
                self._terminate(process)

        except FileNotFoundError:
            self.logger.error("EnergyPlus executable not found.")
            return False

        except Exception:
            self.logger.exception("Running EnergyPlus simulation failed")
            raise

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        """Best-effort cleanup: kill the process if still alive and close its
        stdout pipe so it can't leak as a zombie / dangling file descriptor.

        Safe to call multiple times (idempotent): once the process has exited
        ``poll()`` returns its code and the kill/close calls are no-ops.
        """
        if process.poll() is None:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
        if process.stdout is not None:
            with contextlib.suppress(Exception):
                process.stdout.close()
