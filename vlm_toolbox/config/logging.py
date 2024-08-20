import json
import logging
import sys

from IPython.display import HTML, display
from colorama import Back, Fore, Style

from config.base import BaseConfig

TENSORBOARD_DEFAULT_PORT = 9972
NOTIFY_LEVEL_NUM = 25
logging.addLevelName(NOTIFY_LEVEL_NUM, "NOTIFY")

class LoggerColorsConfig(BaseConfig):
    config = {
        logging.DEBUG: Fore.CYAN,
        NOTIFY_LEVEL_NUM: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT,
    }

    html_config = {
        logging.DEBUG: 'cyan',
        logging.INFO: 'black',
        NOTIFY_LEVEL_NUM: 'blue',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'red; background-color: white; font-weight: bold',
    }

    @staticmethod
    def get_config(logging_number):
        return LoggerColorsConfig.config.get(logging_number, Fore.WHITE)
    
    @staticmethod
    def get_html_config(logging_number):
        return LoggerColorsConfig.html_config.get(logging_number, 'black')

def notify(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTIFY_LEVEL_NUM):
        self._log(NOTIFY_LEVEL_NUM, message, args, **kwargs)

logging.Logger.notify = notify


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, colors=None, **kwargs):
        super().__init__(*args, style='{', **kwargs)
        self.colors = colors if colors else {}

    def format(self, record):
        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL
        message = super().format(record)
        return f"{record.color}{message}{record.reset}"

class JupyterLoggerHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            display(HTML(f"<pre>{msg}</pre>"))
        except Exception:
            self.handleError(record)

class HTMLFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        level_color = LoggerColorsConfig.get_html_config(record.levelno)
        message = super().format(record)
        return f"<span style='color: {level_color};'>{message}</span>"

class FormatterFactory:
    @staticmethod
    def get_handler(notebook=False):
        if notebook:
            handler = JupyterLoggerHandler()
        else:
            handler = logging.StreamHandler(sys.stdout)
        return handler

    @staticmethod
    def get_formatter(notebook=False):
        if notebook:
            return HTMLFormatter('{message}', style='{')
        else:
            return ColoredFormatter(
                '{message}',
                colors={
                    'DEBUG': Fore.CYAN,
                    'INFO': Fore.BLACK if notebook else Fore.WHITE,
                    'NOTIFY': Fore.BLUE,
                    'WARNING': Fore.YELLOW,
                    'ERROR': Fore.RED,
                    'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
                }
            )

class LoggerWrapper:
    def __init__(self, logger, silent=False, notebook=False):
        self.logger = logger
        self.silent = silent
        self.notebook = notebook

    def _process_message(self, message):
        if not self.notebook and isinstance(message, dict):
            return json.dumps(message, indent=4)
        return message

    def info(self, message):
        if not self.silent: self.logger.info(self._process_message(message))
    def debug(self, message):
        if not self.silent: self.logger.debug(self._process_message(message))
    def notify(self, message):
        if not self.silent: self.logger.notify(self._process_message(message))
    def warning(self, message):
        if not self.silent: self.logger.warning(self._process_message(message))
    def error(self, message):
        if not self.silent: self.logger.error(self._process_message(message))
    def critical(self, message):
        if not self.silent: self.logger.critical(self._process_message(message))
    def disable(self):
        self.silent = True
    def enable(self):
        self.silent = False

class LoggerFactory:
    @classmethod
    def create_logger(cls, name, level=logging.DEBUG, notebook=False, silent=False):
        logger = logging.getLogger(name)
        logger.setLevel(level if not silent else logging.NOTSET)
        if not logger.handlers:
            handler = FormatterFactory.get_handler(notebook)
            formatter = FormatterFactory.get_formatter(notebook)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        wrapped_logger = LoggerWrapper(logger, silent)
        return wrapped_logger

def log_tree(d, indent=0, parent_prefix=""):
    items = list(d.items())
    for index, (key, value) in enumerate(items):
        connector = "└── " if index == len(items) - 1 else "├── "
        logger.info(parent_prefix + connector + str(key))
        if isinstance(value, dict):
            new_prefix = parent_prefix + ("    " if index == len(items) - 1 else "│   ")
            log_tree(value, indent + 1, new_prefix)
        else:
            for item in value:
                sub_connector = "└── " if item == list(value)[-1] else "├── "
                logger.info(parent_prefix + ("    " if index == len(items) - 1 else "│   ") + sub_connector + item)

logger = LoggerFactory.create_logger('default logger')
