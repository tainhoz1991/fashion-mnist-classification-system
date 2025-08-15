import argparse
import collections
import torch
import numpy as np
import data_loader.data_loader as module_data
import model.loss_fn as module_loss
import model.model as module_model
import model.metric as module_metric
from config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import os
from dotenv import load_dotenv

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


# config.init_obj function is used to initiate an instance
def main(config):
    logger = config.get_logger('train')

    # initiate data_loader instances from values of config
    data_loader = config.init_obj('data_loader', module_data)

    # create a validation data set
    valid_data_loader = data_loader.split_validation()

    # initiate model architecture instance, then print to console
    model = config.init_obj('model_arch', module_model)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # getattr (x, 'y') = x invokes y = x.y
    loss_fn = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # initiate optimizer instance
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # initiate learning rate scheduler instance if it's configured
    lr_scheduler = None
    if (config['lr_scheduler']["type"]) and (len(config['lr_scheduler']["type"].strip()) > 0):
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, loss_fn, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    # call train() function from base_trainer
    trainer.train()


if __name__ == "__main__":
    load_dotenv()
    # print(f"DATABRICKS_HOST:: {os.getenv('DATABRICKS_HOST')}")

    # Define arguments that program allows
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    # namedtuple is a builtin of collections used to create a class
    # field_names: these are attributes of class being separated by a space " "
    # typename: name of class
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    # build a ConfigParser which keeps all information to initiate classes like such as Model, DataLoader and functions
    # the information are inferred from config.json (it only keeps values for attributes of classes)
    config = ConfigParser.from_args(args, options)
    main(config)