
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
MODEL = 'run_025'
CHECKPOINT_NAME = 'ckpt_best.pth'
PROJECT_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound'
CHECKPOINT = os.path.join(PROJECT_PATH, 'models', MODEL)
EVAL_DIR_KEY = ''
SHOW_IMAGE = False
SAVE_IMAGE = False
DATA_AUGMENTATION = False
# TODO: solve device problem, check behavoir while GPU using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def eval():
    # dataset
    config = cfg.dataset_config
    config['dir_key'] = EVAL_DIR_KEY
    test_dataset = ImageDataset(config, mode=MODE)
    if not DATA_AUGMENTATION:
        config.pop('preprocess_config')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    net = UNet_2d(input_channels=1, num_class=1)
    checkpoint = os.path.join(CHECKPOINT, CHECKPOINT_NAME)
    state_key = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state_key['net'])
    net = net.to(device)
    net.eval()
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    total_dsc = []
    total_precision, total_recall, total_acc = 0, 0, 0
    evaluator = metrics.SegmentationMetrics()
    if len(test_dataloader) == 0:
        raise ValueError('No Data Exist. Please check the data path.')
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
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
            total_precision += evals['precision']
            total_recall += evals['recall']
            total_acc += evals['accuracy']
        
        # TODO: if EVAL_DIR_KEY = ''
        if not os.path.exists(os.path.join(CHECKPOINT, 'images', EVAL_DIR_KEY)):
            os.makedirs(os.path.join(CHECKPOINT, 'images', EVAL_DIR_KEY))

        if SAVE_IMAGE or SHOW_IMAGE:
            ax1.imshow(inputs.cpu()[0,0].detach().numpy(), 'gray')
            ax2.imshow(labels.cpu()[0,0].detach().numpy(), 'gray')
            ax3.imshow(prediction.cpu()[0,0].detach().numpy(), 'gray')
        if SAVE_IMAGE:
            image_code = i + 1
            fig.savefig(os.path.join(CHECKPOINT, 'images', EVAL_DIR_KEY, f'{EVAL_DIR_KEY}_{MODE}_{image_code:04d}.png'))
        if SHOW_IMAGE:
            plt.show()

    mean_dsc = sum(total_dsc)/len(total_dsc)
    std_dsc = [(dsc-mean_dsc)**2 for dsc in total_dsc]
    std_dsc = (sum(std_dsc) / len(std_dsc))**0.5
    print('Precision: {:.3f}'.format((total_precision/len(total_dsc)).item()))
    print('Recall/Sensitivity: {:.3f}'.format((total_recall/len(total_dsc)).item()))
    print('F1 Score: {:.3f}'.format(mean_dsc.item()))
    print('F1 Score std: {:.3f}'.format(std_dsc.item()))

    # TODO: write to txt or excel

if __name__ == "__main__":
    eval()
    