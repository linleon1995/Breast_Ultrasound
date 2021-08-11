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
from cfg import dataset_config
from dataset import dataloader
from utils import train_utils

def dataset_test():
    train_dataset = ImageDataset(dataset_config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    for i, data in enumerate(train_dataloader):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(data['input'][0,0])
        ax2.imshow(data['gt'][0,0])
        plt.show()

def augmentation_test():
    import os
    config_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_aug_512_config.yml'
    config = train_utils.load_config_yaml(config_path)
    config = train_utils.DictAsMember(config)
    path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_aug\{config.dataset.dir_key}'
    if not os.path.exists(path):
        os.mkdir(path)
    dataloader.generate_augment_samples(path, config.aug_samples, config.dataset, mode='train')
    
def precision_test():
    import cv2
    from utils import metrics
    evals = metrics.SegmentationMetrics(['precision'])
    label = cv2.imread(rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\masks\0007.png')
    pred = cv2.imread(rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_30000\result_masks\0007.png')
    pred //= 128
    print(pred.shape)
    plt.imshow(label[...,0], 'gray')
    plt.show()
    plt.imshow(pred[...,2], 'gray')
    plt.show()
    print(evals(label[...,0], pred[...,2])['precision'])

# def path_test():
    # datapath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\archive\\Dataset_BUSI_with_GT"
    # data_analysis(datapath)
    # dataset_test()

    # import cv2
    # path = 'C:\\Users\\test\\Desktop\\Software\\SVN\\Algorithm\\deeplabv3+\\data\\myDataset\\masks_raw\\paintlabel_masks\\0013.PNG'
    # path2 = 'C:\\Users\\test\\Desktop\\Software\\SVN\\Algorithm\\deeplabv3+\\data\\myDataset\\masks\\0013.PNG'
    # # path2 = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\models\run_017'
    # # print(path2)
    # input_image = cv2.imread(path2)
    # plt.imshow(255*input_image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # import os
    # print(os.getcwd())
    # cwd = os.getcwd()
    # print(cwd)
    # print(os.path.join("config", "2dunet_512_config.yaml"))
    # import yaml
    # with open(os.path.join(cwd, "config", "2dunet_512_config.yml"), "r") as stream:
    #     data = yaml.load(stream)
    # print(data)
    
if __name__ == "__main__":
    
    augmentation_test()
    # precision_test()

