import logging
import sys

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.FileHandler('log.txt', mode='w')
handler.setFormatter(formatter)
screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if len(logger.handlers) < 2:
    logger.addHandler(handler)
    logger.addHandler(screen_handler)


def debug(*args, sep=" "):
    logger.debug(sep.join(map(str, args)).replace("\n", "\n" + ' ' * 29))


def info(*args, sep=" "):
    logger.info(sep.join(map(str, args)).replace("\n", "\n" + ' ' * 29))


def warning(*args, sep=" "):
    logger.warning(sep.join(map(str, args)).replace("\n", "\n" + ' ' * 29))


def error(*args, sep=" "):
    logger.error(sep.join(map(str, args)).replace("\n", "\n" + ' ' * 29))
