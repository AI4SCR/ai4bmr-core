import datetime
import logging

VERBOSITY_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}  # default

# ANSI escape codes for various colors
COLORS = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKCYAN": "\033[96m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}

LEVEL_COLORS = {
    logging.DEBUG: COLORS["OKBLUE"],
    logging.INFO: COLORS["OKGREEN"],
    logging.WARNING: COLORS["WARNING"],
    logging.ERROR: COLORS["FAIL"],
    logging.CRITICAL: COLORS["FAIL"],
}


class MyFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, colorful=True, elapsed_time=True):
        super().__init__(fmt, datefmt)
        self.colorful = colorful
        self.elapsed_time = elapsed_time
        self.start_time = datetime.datetime.now()

    def format(self, record):
        record = self.format_colorful(record) if self.colorful else record
        record = self.format_elapsed_time(record) if self.elapsed_time else record
        return super().format(record)

    def format_colorful(self, record):
        levelname = record.levelname
        if record.levelno in LEVEL_COLORS:
            levelname_color = LEVEL_COLORS[record.levelno] + levelname + COLORS["ENDC"]
            record.levelname = levelname_color
        return record

    def format_elapsed_time(self, record):
        record.elapsed_time = str(datetime.datetime.now() - self.start_time)
        return record


def get_logger(
    name: str = "default",
    verbose: int = 0,
    log_to_console=True,
    log_to_file=False,
    filename="ai4bmr_embeddings.log",
    colorful=True,
    elapsed_time=True,
):
    assert (
        log_to_console or log_to_file
    ), "At least one of log_to_console or log_to_file must be True"

    # Determine the logging level based on the verbosity count
    logging_level = VERBOSITY_LEVELS.get(
        verbose, logging.DEBUG
    )  # Default to DEBUG if count exceeds mapping

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    # Remove all handlers associated with the logger (if any)
    logger.handlers.clear()

    # Create formatter
    fmt = f'🕥️%(asctime)s {"[⏳%(elapsed_time)s]" if elapsed_time else ""}🏷️{name} ℹ️ %(levelname)-17s 💬 %(message)s'
    formatter = MyFormatter(
        fmt=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        colorful=colorful,
        elapsed_time=elapsed_time,
    )

    if log_to_console:
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_to_file:
        # add a file handler
        fh = logging.FileHandler(filename)
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
