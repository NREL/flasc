import logging
from datetime import datetime

import coloredlogs


class TracebackInfoFilter(logging.Filter):
    """Clear or restore the exception on log records"""

    def __init__(self, clear=True):
        self.clear = clear

    def filter(self, record):
        if self.clear:
            record._stack_info_hidden, record.stack_info = record.stack_info, None
        elif hasattr(record, "_stack_info_hidden"):
            record.stack_info = record._stack_info_hidden
            del record._stack_info_hidden
        return True


class LoggingManager:
    """
    This class provides easy access to a configured logger.
    """

    def __init__(
        self,
        log_to_console=True,
        console_level="INFO",
        log_to_file=False,
        file_level="INFO",
        console_timestamp=True,
    ):
        self.log_to_console = log_to_console
        self.console_level = console_level
        self.log_to_file = log_to_file
        self.file_level = file_level
        self.console_timestamp = console_timestamp
        self._setup_logger()

    def _setup_logger(self):
        """
        Configures the root logger based on the default or user-specified settings.
        As needed, a StreamHandler is created for console logging or FileHandler
        is created for file logging. Both can be attached to the root
        logger for use throughout FLASC.

        Returns:
            logging.Logger: The root logger from the `logging` module.
        """
        # Create a logger object for flasc
        logger = logging.getLogger(name="flasc")
        logger.setLevel(logging.DEBUG)

        fmt_console = "%(asctime)s %(message)s" if self.console_timestamp else "%(message)s"
        fmt_file = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        file_name = "flasc_{:%Y-%m-%d-%H_%M_%S}.log".format(datetime.now())

        # Remove all existing handlers before adding new ones
        for h in logger.handlers.copy():
            logger.removeHandler(h)

        # Configure and add the console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)
            console_format = coloredlogs.ColoredFormatter(fmt=fmt_console)
            console_handler.setFormatter(console_format)
            console_handler.addFilter(TracebackInfoFilter(clear=True))
            logger.addHandler(console_handler)

        # Configure and add the file handler
        if self.log_to_file:
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(self.file_level)
            file_format = logging.Formatter(fmt_file)
            file_handler.setFormatter(file_format)
            file_handler.addFilter(TracebackInfoFilter(clear=False))
            logger.addHandler(file_handler)

        return logger

    @property
    def logger(self):
        caller_name = f"{type(self).__module__}.{type(self).__name__}"
        return logging.getLogger(caller_name)
