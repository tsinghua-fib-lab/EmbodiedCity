import os
import json
import random

import numpy as np
import torch

from vln.graph_loader import GraphLoader
from vln.clip import PanoCLIP


class DotDict(dict):
    __getattr__ = dict.__getitem__


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(split, dataset_dir):
    data = []
    assert split in ['train', 'test', 'dev']
    with open('%s/data/%s.json' % (dataset_dir, split)) as f:
        for line in f:
            item = dict(json.loads(line))
            data.append(item)
    return data


def get_graph(dataset_dir):
    return GraphLoader(os.path.join(dataset_dir, 'graph')).construct_graph()


def get_pano_clip(clip_cache_dir, image_type='none', allow_cuda=True):
    if image_type == 'none':
        return None
    device = torch.device("cuda" if allow_cuda and torch.cuda.is_available() else "cpu")
    panoCLIP = PanoCLIP(panos_dir='', device=device, fov=60, height=460, width=800, cache_dir=clip_cache_dir)
    return panoCLIP
