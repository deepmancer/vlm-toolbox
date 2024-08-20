from config.base import BaseConfig
from config.path import EXPERIMENTS_ROOT_DIR


class MetricIOConfig(BaseConfig):
    @staticmethod
    def get_config(setup):
        return EXPERIMENTS_ROOT_DIR + setup.get_relative_save_path()
