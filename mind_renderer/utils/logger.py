import inspect
import logging
import os
from enum import Enum
from typing import Optional

from colorama import Fore, ansi
from dotenv import load_dotenv


class Logger:
    _instance = None

    class LoggingLevel(Enum):
        DEBUG = 1
        INFO = 2
        TOOL = 3
        TASK = 4
        THOUGHT_PROCESS = 5
        WARNING = 6
        ERROR = 7
        CRITICAL = 8

        def __lt__(self, other):
            return self.value < other.value

        def __ge__(self, other):
            return self.value >= other.value

        def __le__(self, other):
            return self.value <= other.value

        def __gt__(self, other):
            return self.value > other.value

        def __str__(self) -> str:
            return self.name

        def __repr__(self) -> str:
            return self.name

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, logger_name: str, verbose: bool = True, level: Optional[LoggingLevel] = None):
        if not hasattr(self, "logger"):
            load_dotenv(override=True)
            self.logging_level = level if level else Logger.LoggingLevel[os.getenv("LOGGING_LEVEL", "INFO")]
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(level=self.logging_level.value)
            self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)d)")
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(level=self.logging_level.value)
            self.console_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.console_handler)

    def log(self, message: str, level: LoggingLevel, color: str = ansi.Fore.GREEN) -> None:
        if level >= self.logging_level:
            if len(inspect.stack()) >= 4:
                caller_frame = inspect.stack()[2]
            else:
                caller_frame = inspect.stack()[2]
            caller_name = caller_frame[3]
            caller_line = caller_frame[2]
            message = f"{caller_name}({caller_line}): {message}"
            self.logger.info(color + message + Fore.RESET)

    def debug(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.DEBUG, Fore.BLACK)

    def info(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.INFO, Fore.WHITE)

    def tool_log(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.TOOL, Fore.YELLOW)

    def task_log(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.TASK, Fore.BLUE)

    def thought_process_log(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.THOUGHT_PROCESS, Fore.GREEN)

    def warning(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.WARNING, Fore.YELLOW)

    def error(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.ERROR, Fore.RED)

    def critical(self, message: str) -> None:
        self.log(message, Logger.LoggingLevel.CRITICAL, Fore.MAGENTA)
