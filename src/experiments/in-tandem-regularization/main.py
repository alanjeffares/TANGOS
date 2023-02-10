import os
import torch
import itertools
import argparse
import json
import copy
from pathlib import Path
import numpy as np
import random
import torch.nn as nn
from torch import optim
from sklearn.model_selection import KFold
from src.losses import parameter_schedule, attr_loss, MSE
from src.models import UCI_MLP
from src.regularizers import l1, add_input_noise, mixup_data, mixup_criterion
import src.load_data

import warnings

warnings.filterwarnings("ignore")

EPOCHS = 200
TRAINING_PATIENCE = 30
BATCH_SIZE = 32
K_FOLDS = 5
DEVICE = 'cuda:0'


def train_epoch(model, train_loader, optimiser, loss_func, params, params_ls, regulariser, device='cpu'):
    reg_loss = 0
    for i, (data, label) in enumerate(train_loader):
        model.train()
        data, label = data.to(device), label.to(device)
        if regulariser == 'input_noise':
            data = add_input_noise(data, params['std'])
        optimiser.zero_grad()
        output, _ = model(data)
        pred_loss = loss_func(output, label)

        if (params_ls['lambda_1_curr'] > 0) or (params_ls['lambda_2_curr'] > 0):
            sparsity_loss, correlation_loss = attr_loss(model, data, device=device, subsample=params_ls['subsample'])
            reg_loss = params_ls['lambda_1_curr'] * sparsity_loss + params_ls['lambda_2_curr'] * correlation_loss

        if regulariser == 'l1':
            reg_loss = params['weight'] * l1(model)

        elif regulariser == 'mixup':
            X_mixup, y_a, y_b, lam = mixup_data(data, label, alpha=params['alpha'], device=device)
            output_mixup, _ = model(X_mixup)
            reg_loss = mixup_criterion(loss_func, output_mixup, y_a, y_b, lam)

        loss = pred_loss + reg_loss
        loss.backward()
        optimiser.step()
    return model


def evaluate(model, test_loader, loss_func, device=DEVICE):
    running_loss, running_pred_loss = 0, 0
    for epoch, (data, label) in enumerate(test_loader):
        model.eval()
        data, label = data.to(device), label.to(device)

        output, _ = model(data)

        # compute metric
        pred_loss = loss_func(output, label)
        loss = pred_loss

        running_loss += loss.item()
        running_pred_loss += pred_loss.item()

    return running_pred_loss/(epoch + 1), running_loss/(epoch + 1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def ids_to_dataloader_split(data, train_ids, val_ids, seed):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH_SIZE, sampler=train_subsampler, worker_init_fn=seed_worker, generator=g)
    valloader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH_SIZE, sampler=val_subsampler, worker_init_fn=seed_worker, generator=g)
    return trainloader, valloader


def init_results(tag, seed, datasets, config_regs, latent_sniper_regs, overwrite=False):
    """Helper function to initialise an empty dictionary for storing results"""
    results = load_results(f'experiment_{tag}_seed_{seed}')
    regularisers = config_regs.keys()
    num_combinations_ls = 1
    for i, j in latent_sniper_regs.items():
        num_combinations_ls *= len(j)

    if results and not overwrite:
        raise ValueError('Results already exist, to overwrite pass overwrite = True')
    else:
        results = {}
        for dataset in datasets:
            results[dataset] = {}
            for regulariser in regularisers:
                results[dataset][regulariser] = {}
                num_combinations = 1
                for value in config_regs[regulariser].values():
                    num_combinations *= len(value)
                for i in range(num_combinations * num_combinations_ls):
                    results[dataset][regulariser][i] = {}
                    results[dataset][regulariser][i]['val_loss'] = []

    save_results(results, f'experiment_{tag}_seed_{seed}')

def load_results(file_name):
    curr_dir = os.path.dirname(__file__)
    results_dir = os.path.join(curr_dir, f'results/{file_name}.json')
    file_obj = Path(results_dir)
    if file_obj.is_file():
        with open(results_dir) as f:
            results = json.load(f)
        return results
    else:
        print(f'{file_name}.json not found in results folder, generating new file.')
        return {}

def save_results(results, file_name):
    curr_dir = os.path.dirname(__file__)
    results_dir = os.path.join(curr_dir, f'results/{file_name}.json')
    with open(results_dir, 'w') as f:
        json.dump(results, f)

def run_fold(fold_name, model, trainloader, valloader, config, config_ls, config_dataset, seed):
    tag, dataset, regulariser, params, fold = fold_name.split(':')
    loss_func = MSE if config_dataset['type'] == 'regression' else nn.CrossEntropyLoss()
    l2_weight = config['weight'] if regulariser == 'l2' else 0
    optimiser = optim.Adam(model.parameters(), lr=config_dataset['lr'], weight_decay=l2_weight)
    parameter_scheduler = parameter_schedule(config_ls['lambda_1'], config_ls['lambda_2'], config_ls['param_schedule'])
    best_val_loss = np.inf; last_update = 0
    for epoch in range(EPOCHS):
        lambda_1, lambda_2 = parameter_scheduler.get_reg(epoch)
        config_ls['lambda_1_curr'] = lambda_1
        config_ls['lambda_2_curr'] = lambda_2

        model = train_epoch(model, trainloader, optimiser, loss_func, config, config_ls, regulariser, device=DEVICE)
        val_loss, _ = evaluate(model, valloader, loss_func, device=DEVICE)

        if (val_loss < best_val_loss) or (epoch < 5):
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            last_update = epoch

        # early stopping criteria
        if epoch - last_update == TRAINING_PATIENCE:
            break

    # save best model results for this fold
    results = load_results(f'experiment_{tag}_seed_{seed}')
    results[dataset][regulariser][params]['val_loss'].append(best_val_loss)
    save_results(results, f'experiment_{tag}_seed_{seed}')

    return best_val_loss, best_model, last_update

def run_cv(config_dataset: dict, regulariser: str, params: dict, params_ls: dict, run_name: str, seed: int):
    data_fetcher = getattr(src.load_data, config_dataset['loader'])
    loaders = data_fetcher(seed=0)
    dropout = params['p'] if regulariser == 'dropout' else 0
    batch_norm = True if regulariser == 'batch_norm' else False
    kfold = KFold(n_splits=K_FOLDS, shuffle=False)
    best_loss = np.inf
    # loop through folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(loaders['train'])):
        torch.manual_seed(seed); np.random.seed(seed)
        trainloader, valloader = ids_to_dataloader_split(loaders['train'], train_ids, val_ids, seed=seed)
        fold_name = run_name + f':{fold}'
        model = UCI_MLP(num_features=config_dataset['num_features'], num_outputs=config_dataset['num_outputs'],
                        dropout=dropout, batch_norm=batch_norm).to(DEVICE)
        fold_loss, fold_model, fold_epoch = run_fold(fold_name, model, trainloader, valloader, params, params_ls,
                                                     config_dataset, seed=seed)
        if fold_loss < best_loss:
            best_loss = fold_loss
            best_model = copy.deepcopy(fold_model)
            best_epoch = fold_epoch

    # evalutate best performing model on held out test set
    loss_func = MSE if config_dataset['type'] == 'regression' else nn.CrossEntropyLoss()
    test_loss, _ = evaluate(best_model, loaders['test'], loss_func)
    tag, dataset, regulariser, params = run_name.split(':')
    results = load_results(f'experiment_{tag}_seed_{seed}')
    results[dataset][regulariser][params]['test_loss'] = test_loss
    results[dataset][regulariser][params]['train_final_epoch'] = best_epoch
    print(test_loss)
    save_results(results, f'experiment_{tag}_seed_{seed}')

def grid_search_iterable(parameter_dict: dict) -> list:
    """Generate an iterable list of hyperparameters from a dictionary containing the values to be considered"""
    keys, values = zip(*parameter_dict.items())
    parameter_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return parameter_grid

def load_config(name):
    curr_dir = os.path.dirname(__file__)
    config_dir = os.path.join(curr_dir, f'configs/{name}.json')
    with open(config_dir) as f:
        config_dict = json.load(f)
    config_keys = list(config_dict)
    return config_dict, config_keys

def run_experiment(seeds: list, tag: str):
    # load config files
    config_regs, regularisers = load_config('regularizers')
    config_data, datasets = load_config('datasets')
    latent_sniper_regs = config_regs.pop('TANGOS', None)
    regularisers.remove('TANGOS')
    latent_sniper_iterable = grid_search_iterable(latent_sniper_regs)
    for seed in seeds:
        # initialise results file
        init_results(tag, seed, datasets, config_regs, latent_sniper_regs, overwrite=True)
        for dataset in datasets:
            for regulariser in regularisers:
                parmaeter_iterable = grid_search_iterable(config_regs[regulariser])
                idx = 0
                for param_set in parmaeter_iterable:
                    for param_set_ls in latent_sniper_iterable:
                        run_name = f'{tag}:{dataset}:{regulariser}:{idx}'
                        # run CV on this combination
                        print(run_name)
                        run_cv(config_data[dataset], regulariser, param_set, param_set_ls, run_name, seed)
                        # save record of parameters used for this run
                        param_record = load_results(f'params_record')
                        param_record[f'id_:{seed}:{run_name}'] = param_set
                        save_results(param_record, f'params_record')
                        idx +=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seeds', default=[0], help='Set of seeds to use for experiments')
    parser.add_argument('-tag', default='tag', help='Tag name for set of experiments')
    args = parser.parse_args()
    print(args.seeds)
    run_experiment(seeds=args.seeds, tag=args.tag)