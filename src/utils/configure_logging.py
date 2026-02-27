import logging
import os
import sys

from logging.config import fileConfig


def configure_logging(config_file, verbose, logger_name=None, log_file_suffix=None):
    if config_file and os.path.exists(config_file):
        print(f'Configure logging using {config_file}, logger name: {logger_name}, suffix: {log_file_suffix}')
        fileConfig(config_file)
        if log_file_suffix:
            _override_file_handlers(log_file_suffix)
    else:
        print(f'Configure logging using basic config - verbose: {verbose}, logger name: {logger_name}')
        log_format = '%(asctime)s - %(threadName)s:%(name)s - %(levelname)s - %(message)s'
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(stream=sys.stdout)
                            ])
    return logging.getLogger(logger_name)


def _override_file_handlers(log_file_suffix):
    """Replace the destination of all FileHandlers on all loggers.
    The suffix is inserted before the file extension, e.g.
    segmentation.log -> segmentation-<suffix>.log
    """
    loggers = [logging.root] + list(logging.Logger.manager.loggerDict.values())
    for lgr in loggers:
        if not isinstance(lgr, logging.Logger):
            continue
        for handler in lgr.handlers:
            if isinstance(handler, logging.FileHandler):
                base, ext = os.path.splitext(handler.baseFilename)
                new_path = f'{base}-{log_file_suffix}{ext}'
                handler.close()
                handler.baseFilename = new_path
                handler.mode = 'a'
                handler.stream = handler._open()
