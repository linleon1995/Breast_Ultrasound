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
from dataset.dataloader import data_analysis

def dataset_test():
    train_dataset = ImageDataset(dataset_config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    for i, data in enumerate(train_dataloader):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(data['input'][0,0])
        ax2.imshow(data['gt'][0,0])
        plt.show()

if __name__ == "__main__":
    # datapath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\archive\\Dataset_BUSI_with_GT"
    # data_analysis(datapath)
    # dataset_test()
    import cv2
    path = 'C:\\Users\\test\\Desktop\\Software\\SVN\\Algorithm\\deeplabv3+\\data\\myDataset\\masks_raw\\paintlabel_masks\\0009.PNG'
    path2 = 'C:\\Users\\test\\Desktop\\Software\\SVN\\Algorithm\\deeplabv3+\\data\\myDataset\\masks\\0003.PNG'
    path2 = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\models\run_017'
    print(path2)
    # input_image = cv2.imread(path2)
    # plt.imshow(255*input_image, cmap='gray')
    # plt.show()
