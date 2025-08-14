import json
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch.utils.data import DataLoader


def create_if_not_exist_dir(dir_path: str):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=False)
    return dir_path


def read_json(file_name: str):
    path = Path(file_name)
    with path.open("rt") as file:
        return json.load(file, object_hook=OrderedDict)


def write_json(data, file_name: str):
    path = Path(file_name)
    with path.open("wt") as file:
        json.dump(data, file, indent=4, sort_keys=False)


def inf_loop(data_loader: DataLoader):
    # this function is used to return every batch of a dataloader
    # but after the last batch of dataloader, it will circle back to the first batch and repeat this endlessly
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use: int):
    """
        setup GPU device if available. get gpu device indices which are used for DataParallel
        """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def calculate_mean_std(dataset, batch_size: int = 10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, label in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std
