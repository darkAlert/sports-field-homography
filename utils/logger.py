import sys
import logging
from datetime import date

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(counter=0)
def get_logger(log_path=None, format='%(levelname)s: %(message)s', write_date=True):
    log_name = log_path
    if log_name is not None:
        get_logger.counter += 1
        log_name += str(get_logger.counter)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(stdout_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)

    if write_date:
        logger.info('<<< Logger is created: {} >>>'.format(date.today()))

    return  logger
