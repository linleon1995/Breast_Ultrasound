
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from model import UNet_2d
from torch.utils.data import Dataset, DataLoader
from dataset.dataloader import ImageDataset
from dataset.preprocessing import DataPreprocessing
import numpy as np
from utils import train_utils
from cfg import dataset_config
import os
MODE = 'train'
MODEL = 'run_031'
CHECKPOINT_NAME = 'ckpt_best.pth'
PROJECT_PATH = 'C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\'
CHECKPOINT = os.path.join(PROJECT_PATH, 'models', MODEL)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def eval():
    # dataset
    test_dataset = ImageDataset(dataset_config, mode=MODE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    net = UNet_2d(input_channels=1, num_class=1)
    checkpoint = os.path.join(CHECKPOINT, CHECKPOINT_NAME)
    state_key = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state_key['net'])
    net.eval()
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    total_dsc = []
    for i, data in enumerate(test_dataloader):
        print('Sample: {}'.format(i+1))
        inputs, labels = data['input'], data['gt']
        inputs, labels = inputs.to(device), labels.to(device)
        net = net.to(device)
        # TODO: single evaluation tool
        
        output = net(inputs)
        prediction = torch.round(output)

        tp = ((prediction.data == 1) & (labels.data == 1)).cpu().sum()
        tn = ((prediction.data == 0) & (labels.data == 0)).cpu().sum()
        fn = ((prediction.data == 0) & (labels.data == 1)).cpu().sum()
        fp = ((prediction.data == 1) & (labels.data == 0)).cpu().sum()

        if labels.data.cpu().sum() != 0:
            dsc = 2*tp / (2*tp + fp +fn)
            total_dsc.append(dsc)

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(inputs[0,0].detach().numpy(), 'gray')
        ax2.imshow(labels[0,0].detach().numpy(), 'gray')
        ax3.imshow(prediction[0,0].detach().numpy(), 'gray')
        if not os.path.exists(os.path.join(CHECKPOINT, 'images')):
            os.mkdir(os.path.join(CHECKPOINT, 'images'))
        fig.savefig(os.path.join(CHECKPOINT, 'images', 'img_{}_{:3d}.png'.format(MODE, i)))
        plt.show()
    
    precision =  total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    specificity = total_tn / (total_tn + total_fp)
    f1 = 2*total_tp / (2*total_tp + total_fp + total_fn)
    dsc_mean = sum(total_dsc) / len(total_dsc)
    dsc_std = [(dsc-dsc_mean)**2 for dsc in total_dsc]
    dsc_std = (sum(dsc_std) / len(dsc_std))**0.5

    print('Precision: {:.3f}'.format(precision.item()))
    print('Recall/Sensitivity: {:.3f}'.format(recall.item()))
    print('Specificity: {:.3f}'.format(specificity.item()))
    print('F1 Score: {:.3f}'.format(f1.item()))
    print('Mean DSC: {:.3f}  Std DSC: {:.3f}'.format(dsc_mean.item(), dsc_std.item()))

    # TODO: write to txt or excel

if __name__ == "__main__":
    eval()
    # datapath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\archive\\Dataset_BUSI_with_GT"
    # # data_analysis(datapath)
    # from dataset import dataloader
    # input_data, ground_truth = dataloader.generate_filename_list(datapath, file_key='mask', dir_key='benign')
    # input_data.sort()
    # ground_truth.sort()
    # # print(input_data)
    # print(len(input_data))
    # print('' in datapath)