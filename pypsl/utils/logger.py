# pylint: disable=missing-docstring

import logging
import sys


class Logger:
    """Standardized logging"""

    @staticmethod
    def init():
        logger = Logger.logger()
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)

    @staticmethod
    def logger():
        return logging.getLogger('pypsl')

    @staticmethod
    def debug(msg):
        return Logger.logger().debug(msg)

    @staticmethod
    def info(msg):
        return Logger.logger().info(msg)

    @staticmethod
    def warning(msg):
        return Logger.logger().warning(msg)

    @staticmethod
    def error(msg):
        return Logger.logger().error(msg)


Logger.init()
