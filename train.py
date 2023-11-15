import argparse
import logging
import sys
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from pipeline import Pipeline


def setup_logger():
  logger = logging.getLogger()

  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter(
      fmt='[%(asctime)s][%(levelname)s][%(filename)s:%(funcName)s:%(lineno)d] %(message)s',
      datefmt='%H:%M:%S'
  )
  stream_handler = logging.StreamHandler(stream=sys.stdout)
  file_handler = logging.FileHandler(
      filename=f"log/{datetime.now().strftime('%Y-%m-%d')}.log",
      mode='a'
  )
  stream_handler.setFormatter(formatter)
  file_handler.setFormatter(formatter)
  logger.addHandler(stream_handler)
  logger.addHandler(file_handler)


def main(args):
  cfg = OmegaConf.load(args.config)
  assert isinstance(cfg, DictConfig)

  setup_logger()

  pipeline = Pipeline(cfg)
  pipeline.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('config', type=str, help='Path to a configuration file')
  args = parser.parse_args()
  main(args)
