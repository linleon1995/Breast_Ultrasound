# Classification task on Breast Ultrasound
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cfg import dataset_config
from dataset.dataloader import ImageDataset
# from model import UNet_2d
from utils import train_utils
from dataset import dataset_utils
from utils import configuration
from utils import metrics
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device: {}'.format(device))
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_2dunet_cls_train_config.yml'

logger = train_utils.get_logger('TrainingSetup')


def main():
    # # Configuration
    # # Select device
    # if DEBUG:
    #     config = train_utils.load_config_yaml(CONFIG_PATH)
    # else:
    #     config = train_utils.load_config()
    # config = train_utils.DictAsMember(config)

    # Load and log experiment configuration
    config = configuration.load_config(CONFIG_PATH)
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # Logger
    
    # Dataloader
    train_dataset = ImageDataset(config.dataset, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=config.dataset.shuffle)
    test_dataset_config = config.dataset.copy()
    test_dataset_config.pop('preprocess_config')
    test_dataset = ImageDataset(test_dataset_config, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # create trainer
    default_trainer_builder_class = 'UNetTrainerBuilder'
    trainer_builder_class = config['trainer'].get('builder', default_trainer_builder_class)
    trainer_builder = dataset_utils.get_class(trainer_builder_class, modules=['utils.trainer'])
    trainer = trainer_builder.build(config)
    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()