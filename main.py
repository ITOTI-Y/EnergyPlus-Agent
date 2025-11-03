import time
from pathlib import Path

from src.converter_manager import ConverterManager
from src.runner.runner import EnergyPlusRunner
from src.utils.logging import get_logger, setup_logger

logger_time = time.strftime("%Y%m%d_%H%M%S")
setup_logger(
    level="INFO",
    console_output=True,
    log_file_path=Path(f"./logs/{logger_time}.log"),
)
logger = get_logger(__name__)


if __name__ == "__main__":
    idd_file = Path("./dependencies/Energy+.idd")
    yaml_file = Path("./schemas/building_schema.yaml")
    idf_file_output = Path(f"./output/idf/output_{logger_time}.idf")
    epw_file = Path("./dependencies/Shenzhen.epw")

    manager = ConverterManager(idd_file, yaml_file)
    manager.convert_all()
    manager.save_idf(idf_file_output)

    ep_runner = EnergyPlusRunner(manager._idf)
    ep_runner.run_idf(epw_file_path=epw_file)
