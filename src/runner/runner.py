import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from idfpy import IDF

from src.utils.logging import get_logger


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
    ) -> bool:
        """
        Run EnergyPlus IDF file

        Args:
            idf_file_path: IDF file path
            epw_file_path: EPW weather file path
            output_directory: Output directory, if None, a default directory will be created

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

            output_lines = []
            for line in process.stdout or []:
                line = line.rstrip()
                self.logger.info("[EnergyPlus] {}", line)
                output_lines.append(line)

            return_code = process.wait()

            if return_code != 0:
                self.logger.error("EnergyPlus exited with code {}", return_code)
                return False

            self.logger.info("EnergyPlus simulation completed successfully.")
            return True

        except FileNotFoundError:
            self.logger.error("EnergyPlus executable not found.")
            return False

        except Exception:
            self.logger.exception("Running EnergyPlus simulation failed")
            raise
