import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.data.dictionary import create_dictionaries
from model_train import create_datasets, create_model
from model.training import settings

if __name__ == '__main__':

    # print('Start')
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='give the path to the folder that contains the saved_model')
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    args = parser.parse_args()

    settings.device = args.device
    path = args.path
    with open(path + '/config_file.json') as f:
        config = json.load(f)

    if 'path' not in config:
        print('set path=', Path(args.config_file).parent)
        config['path'] = Path(args.config_file).parent

    if args.path is not None:
        print('WARNING: setting path to {}'.format(args.path))
        config['path'] = args.path
        config['model']['path'] = args.path

    path_parent = os.path.dirname(os.getcwd())

    # dictionairies containing the characters, entities, etc.
    dictionaries = create_dictionaries(config, 'predict')
    datasets, data, evaluate = create_datasets(config, dictionaries)
    model, parameters = create_model(config, dictionaries)

    device = torch.device(settings.device)
    model = model.to(device)

    collate_fn = model.collate_func(datasets, device)
    batch_size = config['optimizer']['batch_size']

    filename = os.path.join(path, 'savemodel.pth')
    model.load_state_dict(torch.load(filename))

    metrics = {name: model.create_metrics() for name in config['trainer']['evaluate']}

    # evaluate
    for name in config['trainer']['evaluate']:
        print('Start evaluating', name)

        loader = DataLoader(datasets[name], collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

        model.eval()

        if hasattr(model, 'begin_epoch'):
            model.begin_epoch()

        with open(config['model']['path'] + '/{}.json'.format(name), 'w') as file:
            for i, minibatch in enumerate(tqdm(loader)):
                predictions = model.predict(**minibatch)
                for pred in predictions[1]:
                    json.dump(pred, file)
                    file.write('\n')

        if hasattr(model, 'end_epoch'):
            model.end_epoch(name)
