import inspect
import logging
import os
import time
from enum import Enum
from typing import Optional

from colorama import Fore, ansi
from dotenv import load_dotenv

from mind_renderer.utils.config_loader import ConfigLoader


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

    def __init__(
        self, logger_name: str, parent_folder: str = "", verbose: bool = True, level: Optional[LoggingLevel] = None
    ):
        if not hasattr(self, "logger"):
            load_dotenv(override=True)
            self.config = ConfigLoader()
            self.logging_level = level if level else Logger.LoggingLevel[self.config.get_value("log_level", "INFO")]
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(level=self.logging_level.value)

            # Remove all existing handlers
            if self.logger.hasHandlers():
                self.logger.handlers.clear()

            self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)d)")
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(level=self.logging_level.value)
            self.console_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.console_handler)

            # File handler
            if parent_folder:
                log_folder_root = self.config.get_value("logger.folder_root", "logs")
                log_folder = os.path.join(parent_folder, log_folder_root)
                if not os.path.exists(log_folder):
                    os.makedirs(log_folder)
                timestamp = str(int(time.time()))
                log_file = os.path.join(log_folder, f"{timestamp}-{logger_name}.log")
                self.file_handler = logging.FileHandler(log_file)
                self.file_handler.setLevel(level=self.logging_level.value)
                self.file_handler.setFormatter(self.formatter)
                self.logger.addHandler(self.file_handler)

    def log(self, message: str, level: LoggingLevel, color: str = ansi.Fore.GREEN, write_to_file: bool = False) -> None:
        if level >= self.logging_level:
            if len(inspect.stack()) >= 4:
                caller_frame = inspect.stack()[3]
            else:
                caller_frame = inspect.stack()[2]
            caller_name = caller_frame.function
            caller_line = caller_frame.lineno
            formatted_message = f"{caller_name}({caller_line}): {message}"

            # Console logging
            console_message = color + formatted_message + Fore.RESET
            self.console_handler.handle(
                logging.LogRecord(
                    name=self.logger.name,
                    level=level.value,
                    pathname=caller_frame.filename,
                    lineno=caller_line,
                    msg=console_message,
                    args=None,
                    exc_info=None,
                )
            )

            # File logging
            if write_to_file:
                self.file_handler.handle(
                    logging.LogRecord(
                        name=self.logger.name,
                        level=level.value,
                        pathname=caller_frame.filename,
                        lineno=caller_line,
                        msg=formatted_message,
                        args=None,
                        exc_info=None,
                    )
                )

    def debug(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.DEBUG, Fore.BLACK, write_to_file)

    def info(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.INFO, Fore.WHITE, write_to_file)

    def tool_log(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.TOOL, Fore.YELLOW, write_to_file)

    def task_log(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.TASK, Fore.BLUE, write_to_file)

    def thought_process_log(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.THOUGHT_PROCESS, Fore.GREEN, write_to_file)

    def warning(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.WARNING, Fore.YELLOW, write_to_file)

    def error(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.ERROR, Fore.RED, write_to_file)

    def critical(self, message: str, write_to_file: bool = False) -> None:
        self.log(message, Logger.LoggingLevel.CRITICAL, Fore.MAGENTA, write_to_file)
