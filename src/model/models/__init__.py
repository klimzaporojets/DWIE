from model.models.main_model import MainModel

models = {}


def register_model(name, factory):
    models[name] = factory


def model_create(config, dictionaries):
    name = config['name']
    if name in models:
        return models[name](dictionaries, config)
    else:
        raise BaseException('no such model:', name)


register_model('main_model', MainModel)
