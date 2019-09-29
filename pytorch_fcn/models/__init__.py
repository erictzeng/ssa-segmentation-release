from .fcn8s import VGG16_FCN8s
from .drn import drn_c_26


models = {
    'fcn8s': VGG16_FCN8s,
    'drn': lambda **kwargs: drn_c_26(pretrained=True, finetune=True, out_map=True, **kwargs),
}


def get_model(name, *args, **kwargs):
    return models[name](*args, **kwargs)
