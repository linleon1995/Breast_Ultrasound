
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cfg
from model import UNet_2d
from dataset.dataloader import ImageDataset
from dataset.preprocessing import DataPreprocessing
from utils import train_utils
from utils import metrics
MODE = 'test'
MODEL = 'run_034'
CHECKPOINT_NAME = 'ckpt_best.pth'
PROJECT_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound'
CHECKPOINT = os.path.join(PROJECT_PATH, 'models', MODEL)
EVAL_DIR_KEY = 'benign'
SHOW_IMAGE = True
SAVE_IMAGE = False
DATA_AUGMENTATION = True
# TODO: solve device problem, check behavoir while GPU using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_aug_512_config.yml'

def eval():
    # dataset
    # config = cfg.dataset_config
    config = train_utils.load_config_yaml(CONFIG_PATH)
    config = train_utils.DictAsMember(config)
    dataset_config = config.dataset
    dataset_config['dir_key'] = EVAL_DIR_KEY
    test_dataset = ImageDataset(dataset_config, mode=MODE)
    if not DATA_AUGMENTATION:
        dataset_config.pop('preprocess_config')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    net = UNet_2d(input_channels=1, num_class=1)
    checkpoint = os.path.join(CHECKPOINT, CHECKPOINT_NAME)
    state_key = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state_key['net'])
    net = net.to(device)
    net.eval()
    total_precision, total_recall, total_dsc = [], [], []
    # total_tp, total_fp, total_fn = 0, 0, 0
    evaluator = metrics.SegmentationMetrics()
    if len(test_dataloader) == 0:
        raise ValueError('No Data Exist. Please check the data path or data_plit.')
    # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
    for i, data in enumerate(test_dataloader):
        print('Sample: {}'.format(i+1))
        inputs, labels = data['input'], data['gt']
        inputs, labels = inputs.to(device), labels.to(device)
        output = net(inputs)
        prediction = torch.round(output)

        evals = evaluator(labels, prediction)
        tp, fp, fn = evaluator.tp, evaluator.fp, evaluator.fn
        if (2*tp + fp + fn) != 0:
            total_dsc.append(evals['f1'])

        total_precision.append(evals['precision'])
        total_recall.append(evals['recall'])
        # total_tp += tp
        # total_fp += fp
        # total_fn += fn

        # TODO: if EVAL_DIR_KEY = ''
        if not os.path.exists(os.path.join(CHECKPOINT, 'images', EVAL_DIR_KEY)):
            os.makedirs(os.path.join(CHECKPOINT, 'images', EVAL_DIR_KEY))

        if SAVE_IMAGE or SHOW_IMAGE:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
            ax1.imshow(inputs.cpu()[0,0].detach().numpy(), 'gray')
            ax2.imshow(labels.cpu()[0,0].detach().numpy(), 'gray')
            ax3.imshow(prediction.cpu()[0,0].detach().numpy(), 'gray')
        if SAVE_IMAGE:
            image_code = i + 1
            fig.savefig(os.path.join(
                CHECKPOINT, 'images', EVAL_DIR_KEY, f'{EVAL_DIR_KEY}_{MODE}_{image_code:04d}.png'))
            plt.close(fig)
        if SHOW_IMAGE:
            plt.show()
    # print(tp, evaluator.total_tp, total_tp)

    precision = metrics.precision(evaluator.total_tp, evaluator.total_fp).item() if evaluator.total_tp != 0 else 0
    recall = metrics.recall(evaluator.total_tp, evaluator.total_fn).item() if evaluator.total_tp != 0 else 0
    print(30*'-')
    print(f'total precision: {precision:.3f}')
    print(f'total recall: {recall:.3f}\n')

    mean_precision = sum(total_precision)/len(total_precision)
    mean_recall = sum(total_recall)/len(total_recall)
    mean_dsc = sum(total_dsc)/len(total_dsc) if len(total_dsc) != 0 else 0
    if sum(total_dsc) == 0:
        std_dsc = 0
    else:
        std_dsc = [(dsc-mean_dsc)**2 for dsc in total_dsc]
        std_dsc = ((sum(std_dsc) / len(std_dsc))**0.5).item()
    print(f'mean precision: {mean_precision:.3f}')
    print(f'mean recall: {mean_recall:.3f}')
    print(f'mean dsc: {mean_dsc:.3f}')
    print(f'std dsc: {std_dsc:.3f}')

    # TODO: write to txt or excel

if __name__ == "__main__":
    eval()
    