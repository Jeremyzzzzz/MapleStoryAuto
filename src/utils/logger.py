'''
Global Logger
'''
# Standard Import
import logging
import datetime
import os

class MSLogger:
    '''
    MapleStory AutoBot Logger
    '''
    def __init__(self, name="MSBot"):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        os.makedirs("log", exist_ok=True)
        file_handler = logging.FileHandler(f"log/{name}_{now_str}.log", mode='w', encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

        self._file_handler = file_handler
        self._console_handler = console_handler

    def set_level(self, level):
        '''
        Set logger level, e.g. DEBUG, INFO, WARNING
        '''
        self._logger.setLevel(level)
        self._file_handler.setLevel(level)
        self._console_handler.setLevel(level)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def addHandler(self, handle):
        self._logger.addHandler(handle)

# Initialize shared logger instance for global import
logger = MSLogger()
