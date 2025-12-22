import time
from pathlib import Path
from typing import Literal

from typer import Typer

from src.converter_manager import ConverterManager
from src.runner.runner import EnergyPlusRunner
from src.utils.logging import get_logger, setup_logger
from src.validator.data_model import BaseSchema

logger_time = time.strftime("%Y%m%d_%H%M%S")
setup_logger(
    level="INFO",
    console_output=True,
    log_file_path=Path(f"./logs/{logger_time}.log"),
)
logger = get_logger(__name__)

app = Typer()

BaseSchema.set_idf(Path("./dependencies/Energy+.idd"))


@app.command()
def convert_idf():
    idd_file = Path("./dependencies/Energy+.idd")
    yaml_file = Path("./schemas/building_schema.yaml")
    idf_file_output = Path(f"./output/idf/output_{logger_time}.idf")
    epw_file = Path("./dependencies/Shenzhen.epw")
    manager = ConverterManager(idd_file, yaml_file)
    manager.convert_all()
    manager.save_idf(idf_file_output)
    ep_runner = EnergyPlusRunner(manager._idf)
    ep_runner.run_idf(epw_file_path=epw_file)


@app.command()
def mcp_server(
    transport: Literal["stdio", "http", "sse", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    from src.mcp.server import mcp

    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, port=port, host=host)


if __name__ == "__main__":
    app()
