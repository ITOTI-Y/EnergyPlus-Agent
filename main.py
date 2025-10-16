from pathlib import Path
import time

from src.converter_manager import ConverterManager
from src.utils.logging import setup_logger, get_logger
from src.runner.runner import EnergyPlusRunner

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
    idf_file_output = Path(f"./output/output_{logger_time}.idf")
    epw_file = Path("./dependencies/Shenzhen.epw")

    manager = ConverterManager(idd_file, yaml_file)
    idf = manager.convert_all()
    manager.save_idf(idf_file_output)
    
    EP_run = EnergyPlusRunner(idd_file)
    run_result = EP_run.run_idf(idf_file_output, epw_file)
