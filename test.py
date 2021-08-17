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
import cv2
import os


def dataset_test():
    train_dataset = ImageDataset(dataset_config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    for i, data in enumerate(train_dataloader):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(data['input'][0,0])
        ax2.imshow(data['gt'][0,0])
        plt.show()

def augmentation_test():
    
    config_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_aug_512_config.yml'
    config = train_utils.load_config_yaml(config_path)
    config = train_utils.DictAsMember(config)
    path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_aug\{config.dataset.dir_key}'
    if not os.path.exists(path):
        os.mkdir(path)
    dataloader.generate_augment_samples(path, config.aug_samples, config.dataset, mode='train')
    

def save_file_names():
    with open(rf'C:\Users\test\Desktop\Leon\Projects\Defect_Segmentation\s.txt', 'a+') as fw:
        for i in range(1, 576):
            fw.write(f'{i:04d}\n')

def save_file_names():
    with open(rf'C:\Users\test\Desktop\Leon\Projects\Defect_Segmentation\s.txt', 'a+') as fw:
        for i in range(1, 576):
            fw.write(f'{i:04d}\n')

def pred_and_label():
    image_code = '0013.png'
    pred_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_25000\result_masks'
    pred_path = os.path.join(pred_path, image_code)
    label_path = rf'C:\Users\test\Desktop\Leon\Datasets\DAGM\DAGM_assignment_split\formal_trial\masks'
    label_path = os.path.join(label_path, image_code)
    pred = cv2.imread(pred_path)[...,2]
    pred //= 128
    label = cv2.imread(label_path)[...,2]
    # plt.imshow(np.float32(pred+label*2), 'gray')
    plt.imshow(np.float32(pred*2+label), 'gray')
    plt.show()

def image_and_label():
    i = 13
    image_code = f'{i:04d}.png'
    # image_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\images'
    # label_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\masks'
    
    image_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\images'
    label_path = rf'C:\Users\test\Desktop\Leon\Datasets\DAGM\DAGM_assignment_split\formal_trial\masks'
    image_path = os.path.join(image_path, image_code)
    label_path = os.path.join(label_path, image_code)
    # print(image_path, label_path)
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)
    # print(image.shape)
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(image, 'gray')
    ax2.imshow(label*255, 'gray')
    plt.title(i)
    plt.show()

def precision_test():
    
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

def convert_DAGM_mask_main():
    # load_path = rf'C:\Users\test\Desktop\Leon\Datasets\DAGM\DAGM_KaggleUpload\Class2\Test\Label'
    load_path = rf'C:\Users\test\Desktop\Leon\Datasets\DAGM\DAGM_KaggleUpload\Class2\Train\Label'
    save_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\masks'
    convert_DAGM_mask(load_path, save_path)

def convert_DAGM_mask(load_path, save_path):
    for idx, f in enumerate(os.listdir(load_path)):
        if f.split('.')[1] in ['png', 'PNG', 'jpg', 'jpeg']:
            print(f'step {idx+1}')
            mask = cv2.imread(os.path.join(load_path, f))
            mask = convert_mask(mask)
            save_name = f.split('_label')[0] + f.split('_label')[1]
            # print(f, save_name)
            # plt.imshow(np.float32(mask), 'gray')
            # plt.show()
            cv2.imwrite(os.path.join(save_path, save_name), mask)

def convert_mask(image):
    image = image // np.max(image) if np.max(image) > 0 else image
    return image

# def path_test():
    # datapath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\archive\\Dataset_BUSI_with_GT"
    # data_analysis(datapath)
    # dataset_test()

    # 
    # path = 'C:\\Users\\test\\Desktop\\Software\\SVN\\Algorithm\\deeplabv3+\\data\\myDataset\\masks_raw\\paintlabel_masks\\0013.PNG'
    # path2 = 'C:\\Users\\test\\Desktop\\Software\\SVN\\Algorithm\\deeplabv3+\\data\\myDataset\\masks\\0013.PNG'
    # # path2 = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\models\run_017'
    # # print(path2)
    # input_image = cv2.imread(path2)
    # plt.imshow(255*input_image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # 
    # print(os.getcwd())
    # cwd = os.getcwd()
    # print(cwd)
    # print(os.path.join("config", "2dunet_512_config.yaml"))
    # import yaml
    # with open(os.path.join(cwd, "config", "2dunet_512_config.yml"), "r") as stream:
    #     data = yaml.load(stream)
    # print(data)
    
if __name__ == "__main__":
    
    # augmentation_test()
    # precision_test()
    pred_and_label()
    # save_file_names()
    # image_and_label()
    # convert_DAGM_mask_main()

