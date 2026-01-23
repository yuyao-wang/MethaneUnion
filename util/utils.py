import argparse
import os
import pathlib
import yaml

def load_config(path=None):
    if path is None:
        path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'configs/dp_config.yaml')
        
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dp_config.yaml')
    args = parser.parse_args()
    return args