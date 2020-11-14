import argparse
import gin
from gin_tune import tune_gin
from fault_tolerance.helpers import get_gin_config
import ray
import logging


parser = argparse.ArgumentParser("Run an experiment")
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--n_cpus', type=int, default=None)
parser.add_argument('--n_gpus', type=int, default=None)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    ray.init(num_cpus=args.n_cpus, num_gpus=args.n_gpus)
    gin.parse_config(get_gin_config(args.config))
    tune_gin()