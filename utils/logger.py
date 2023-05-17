import logging
import os


class Logger():
    """Define a logger for the training process.

    Args:
        output_dir (str): The directory to save the log files.
        log_file (str): The name of the log file.
    """
    _handle = None
    _root = None

    @staticmethod
    def init(output_dir, log_file):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        format = '[%(asctime)s %(filename)s:%(lineno)d %(levelname)s {}] ' \
                 '%(message)s'.format(log_file)
        log_path = os.path.join(output_dir, f'{log_file}.log')
        logging.basicConfig(
            filename=log_path, level=logging.INFO, format=format)

        Logger._handle = logging.FileHandler(log_path)
        Logger._root = logging.getLogger()

    @staticmethod
    def enable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.addHandler(Logger._handle)

    @staticmethod
    def disable_file():
        if Logger._handle is None or Logger._root is None:
            raise Exception('Invoke Logger.init() first!')
        Logger._root.removeHandler(Logger._handle)
