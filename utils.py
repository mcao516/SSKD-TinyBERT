import json
import torch
import random
import numpy as np


def set_random_seed(args, n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_cluster_map(args):
    cluster_map = None
    if args.cluster_map_path is not None:
        with open(args.cluster_map_path) as f:
            cluster_map = json.load(f)
    return cluster_map