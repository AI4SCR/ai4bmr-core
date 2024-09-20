from ai4bmr_core.utils.logging import get_logger


def test_default():
    logger = get_logger("Default")
    logger.info("Default info message")
    logger.debug("Default debug message")
    logger.warning("Default warning message")
    logger.error("Default error message")
    logger.critical("Default critical message")


def test_info():
    logger = get_logger("Info", verbose=1)
    logger.info("Info info message")
    logger.debug("Info debug message")
    logger.warning("Info warning message")
    logger.error("Info error message")
    logger.critical("Info critical message")


def test_debug():
    logger = get_logger("Debug", verbose=2, elapsed_time=False)
    logger.info("Debug info message")
    logger.debug("Debug debug message")
    logger.warning("Debug warning message")
    logger.error("Debug error message")
    logger.critical("Debug critical message")
