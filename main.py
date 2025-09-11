from pathlib import Path
import time

from src.converters.converter_manager import ConverterManager
from src.utils.logging import setup_logger, get_logger

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

    manager = ConverterManager(idd_file, yaml_file)
    idf = manager.convert_all()