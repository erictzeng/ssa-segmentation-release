import os.path

from .cityscapes import Cityscapes
from .cyclegan import CycleGTA5
from .gta5 import GTA5


datasets = {
    'cityscapes': Cityscapes,
    'gta5': GTA5,
    'cyclegta5': CycleGTA5,
}


def get_dataset(name, *args, **kwargs):
    return datasets[name](os.path.join('data', name), *args, **kwargs)
