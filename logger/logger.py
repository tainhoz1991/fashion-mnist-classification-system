import logging
import logging.config
from pathlib import Path
from utils import read_json


def configure_logging(save_dir, log_config_path="logger/logger_configuration.json", default_level=logging.INFO):
    log_config = Path(log_config_path)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
