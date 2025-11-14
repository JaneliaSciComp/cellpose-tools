import logging
import os
import sys

from logging.config import fileConfig


def configure_logging(config_file, verbose, logger_name=None):
    if config_file and os.path.exists(config_file):
        print(f'Configure logging using {config_file}, logger name: {logger_name}')
        fileConfig(config_file)
    else:
        print(f'Configure logging using basic config - verbose: {verbose}, logger name: {logger_name}')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(stream=sys.stdout)
                            ])
    return logging.getLogger(logger_name)
