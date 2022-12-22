import numpy as np
from pydoc import locate
import yaml

def create_data_from_serialization(data_from_yaml):
    if isinstance(data_from_yaml, dict):
        for k, v in data_from_yaml.items():
            data_from_yaml[k] = create_data_from_serialization(data_from_yaml[k])
        if 'type' in data_from_yaml.keys():
                Class = locate(data_from_yaml['type'])
                del data_from_yaml['type']
                return Class(**data_from_yaml)
    if isinstance(data_from_yaml, list):
        for idx in range(len(data_from_yaml)):
            data_from_yaml[idx] = create_data_from_serialization(data_from_yaml[idx])
    return data_from_yaml


def create_data_from_yaml(yaml_path):
    with open(yaml_path, 'r') as outfile:
        data = yaml.load(outfile, Loader=yaml.Loader)
    return create_data_from_serialization(data)