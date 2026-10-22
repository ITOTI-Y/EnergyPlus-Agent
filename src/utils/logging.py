import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")


def setup_logger(
    level: str = "INFO",
    format_str: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    log_file_path: Path | None = None,
) -> None:
    """Send logs to stderr and, when ``log_file_path`` is given, to that file.

    Args:
        level: Minimum log level.
        format_str: Loguru format with color tags.
        log_file_path: File that also receives the logs.
    """
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=format_str, colorize=True)
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file_path, level=level.upper(), format=format_str, encoding="utf-8"
        )
