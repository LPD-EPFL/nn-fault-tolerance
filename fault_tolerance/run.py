import argparse
import gin
from gin_tune import tune_gin
from fault_tolerance.helpers import get_gin_config


parser = argparse.ArgumentParser("Run an experiment")
parser.add_argument('--config', type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    gin.parse_config(get_gin_config(args.config))
    tune_gin()