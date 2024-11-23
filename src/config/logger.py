import json
import logging
import sys
from typing import Union

from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info, ansi=True).log(
            level, record.getMessage()
        )

def format_record(record: dict) -> str:
    format_string = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level}</level> | "
    )

    if "extra" in record and record["extra"]:
        for binding_key, binding_value in record["extra"].items():
            if binding_key.startswith("custom_bind_"):
                format_string += f"<blue><b>{binding_key[12:].upper()}: {binding_value!r}</b></blue> | "

    format_string += "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"

    if record.get('exception', None) is not None:
        format_string += "{exception}\n"

    if "payload" in record["extra"]:
        try:
            record["extra"]["payload"] = json.dumps(record["extra"]["payload"], indent=4, default=str)
            format_string += "<level>Payload: {extra[payload]}</level>\n"
        except (TypeError, OverflowError) as e:
            format_string += f"<level>Payload could not be serialized: {e}</level>\n"

    return format_string

def init_logging(level: Union[int, str] = logging.INFO):
    logger.remove()

    root_logger = logging.getLogger()
    root_logger.setLevel(level=level)
    
    intercept_handler = InterceptHandler()
    logging.basicConfig(handlers=[intercept_handler], level=level, force=True)

    uvicorn_loggers = (
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("uvicorn.") or True
    )

    for uvicorn_logger in uvicorn_loggers:
        uvicorn_logger.handlers = [intercept_handler]

    logger.configure(
        handlers=[
            {"sink": sys.stdout, "level": level, "colorize": True, "format": format_record},
        ]
    )

def get_logger(**binding_params):
    if not binding_params:
        return logger
    return logger.bind(**{f"custom_bind_{k}": v for k, v in binding_params.items()})

__all__ = ["init_logging", "get_logger"]
