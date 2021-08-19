
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from cfg import DATA_PATH
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from dataset import preprocessing
from dataset import dataset_utils

# TODO: General solution
def generate_filename_list(path, file_key, dir_key='', only_filename=False):
    input_paths, gt_paths = [], []
    for root, dirs, files in os.walk(path):
        for f in files:
            if not only_filename:
                fullpath = os.path.join(root, f)
            else:
                fullpath = f
            if dir_key in fullpath:
                if file_key in fullpath:
                    gt_paths.append(fullpath)
                else:
                    input_paths.append(fullpath)
    return input_paths, gt_paths

def data_analysis(path, dir_key, file_key):
    """check image and mask value range"""
    input_paths, gt_paths = generate_filename_list(path, file_key, dir_key)
    # print(len(input_paths), len(gt_paths))
    # assert len(input_paths) == len(gt_paths)
    image_height, image_width = [], []
    for path, gt_path in zip(input_paths, gt_paths):
        image = cv2.imread(path)
        gt = cv2.imread(gt_path)
        # assert image.shape() == gt.shape()
        image_height.append(image.shape[0])
        image_width.append(image.shape[1])
    print("Height Information (Min: {}, Max:{} Mean:{} Std:{})".
        format(min(image_height), max(image_height), sum(image_height)/len(image_height), sum(image_height)/len(image_height)))
    print("Width Information (Min: {}, Max:{} Mean:{} Std:{})".
        format(min(image_width), max(image_width), sum(image_width)/len(image_width), sum(image_width)/len(image_width)))

# TODO: Rewrite
# TODO: general data analysis tool
def data_preprocessing(path, file_key, dir_key):
    """merge mask"""
    input_paths, gt_paths = generate_filename_list(path, file_key, dir_key)
    for idx, gt_path in enumerate(gt_paths):
        if 'mask_1' in gt_path:
            print(idx)
            keyword = gt_path.split('mask_')[0]
            start = min(0, idx-2)
            end = max(len(gt_paths)-1, idx+2)
            mask_path_list = []
            for i in range(start, end):
                if keyword in gt_paths[i]:
                    mask_path_list.append(gt_paths[i])
            # Merge masks and save
            mask = 0
            for mask_path in mask_path_list:
                mask = mask + cv2.imread(mask_path)
                os.remove(mask_path)
            filename = mask_path_list[0]
            cv2.imwrite(filename, mask)


def generate_augment_samples(path, aug_config, dataset_config, mode):
    dataset = ImageDataset(dataset_config, mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crop_size = aug_config.crop_size
    ratio = 0.75
    for idx, data in enumerate(dataloader):
        print(f'Sample {idx}')
        inputs, labels = data['input'].numpy(), data['gt'].numpy()
        # inputs, labels = inputs.to(device), labels.to(device)
        height, width = inputs.shape[2:]
        input_list = [inputs[0,0]]
        label_list = [labels[0,0]]

        # flipping
        if aug_config.flip:
            flipped_image, flipped_label = [], []
            for image, label in zip(input_list, label_list):
                flip_image, flip_label = preprocessing.rand_flip(image, label, 1.0)
                flipped_image.append(flip_image)
                flipped_label.append(flip_label)
            input_list.extend(flipped_image)    
            label_list.extend(flipped_label)    

        # scaling
        # cropping
        if aug_config.crop:
            cropped_image, cropped_label = [], []
            for image, label in zip(input_list, label_list):
                def crop5(image):
                    crop_dict = {}
                    crop_dict['left_top'] = image[:crop_size, :crop_size]

                    if crop_size/width < ratio and crop_size/height < ratio:
                        crop_dict['right_bot'] = image[(height-crop_size):, (width-crop_size):]

                        offset_height = (height - crop_size) // 2
                        offset_width = (width - crop_size) // 2
                        crop_dict['center'] = image[offset_height:height-offset_height, offset_width:width-offset_width]

                    if crop_size/width < ratio:
                        crop_dict['right_top'] = image[:crop_size, (width-crop_size):]

                    if crop_size/height < ratio:
                        crop_dict['left_bot'] = image[(height-crop_size):, :crop_size]
                    crop_dict = list(crop_dict.values())
                    return crop_dict
                # TODO: use dict instead of list
                cropped_image.extend(crop5(image))
                cropped_label.extend(crop5(label))
                    

                # cropped_image.append(image[:height, :width])
                # cropped_image.append(image[:height, :width])
                # cropped_label.append(label[:height, :width])
            input_list = cropped_image
            label_list = cropped_label
        j = 1
        for image, label in zip(input_list, label_list):
            cv2.imwrite(os.path.join(path, f'{dataset_config.dir_key}_{idx:04d}_aug{j}.png'), image*255)
            cv2.imwrite(os.path.join(path, f'{dataset_config.dir_key}_{idx:04d}_mask_aug{j}.png'), label*255)
            j += 1
        # plt.imshow(label_list[-1], 'gray')
        # plt.show()
        

# TODO: test execute time between generate path list and load local file
# TODO: imread for RGB, Gray?
class ImageDataset(Dataset):
    def __init__(self, dataset_config, mode):
        data_split = dataset_config['data_split']
        assert isinstance(data_split, list)
        assert isinstance(data_split[0], float)
        assert isinstance(data_split[1], float)
        assert data_split[0] + data_split[1] == 1
        self.data_split = data_split
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_config = dataset_config
        self.dir_key = self.dataset_config['dir_key']
        self.file_key = self.dataset_config['file_key']

        # # Split training and testing dataset
        # input_data, ground_truth = generate_filename_list(self.dataset_config['data_path'], self.file_key, self.dir_key)
        # input_data.sort()
        # ground_truth.sort()
        # split = int(len(input_data)*self.data_split[0])
        # if mode == 'train':
        #     self.input_data, self.ground_truth = input_data[:split], ground_truth[:split]
        # elif mode == 'test':
        #     self.input_data, self.ground_truth = input_data[split:], ground_truth[split:]

        if mode == 'train':
            data_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\train.txt'
        elif mode == 'test':
            data_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\valid.txt'
        self.input_data = dataset_utils.load_content_from_txt(data_path)
        self.ground_truth = [f.split('.png')[0]+'_mask.png' for f in self.input_data] 
        print("{}  Samples: {}".format(self.mode, len(self.input_data)))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        # Load images
        input_image = cv2.imread(self.input_data[idx])[...,0:1]
        gt_image = cv2.imread(self.ground_truth[idx])[...,0:1]

        # TODO: params: pooling_size
        # Data preprocessing
        pooling_size = 16
        H = input_image.shape[0]
        W = input_image.shape[1]
        top, left = (pooling_size-H%pooling_size)//2, (pooling_size-W%pooling_size)//2
        bottom, right = (pooling_size-H%pooling_size)-top, (pooling_size-W%pooling_size)-left
        input_image = cv2.copyMakeBorder(input_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)
        if gt_image is not None:
            gt_image = cv2.copyMakeBorder(gt_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)

        if 'preprocess_config' in self.dataset_config:
            preprocessor = preprocessing.DataPreprocessing(self.dataset_config['preprocess_config'])
            input_image, gt_image = preprocessor(input_image, gt_image)
            
        # TODO: remove judgement after do experiment
        # print(input_image.dtype, gt_image.dtype)
        # normalization = True
        # if normalization:
        #     input_image = preprocessing.z_score_normalize(input_image)
        #     if gt_image is not None:
        #         gt_image = preprocessing.z_score_normalize(gt_image)
        # print(input_image.dtype, gt_image.dtype)

        # Transform to Torch tensor
        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)


        return {'input': input_image, 'gt': gt_image}
  

if __name__ == "__main__":
    datapath = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\archive\\Dataset_BUSI_with_GT"
    # data_analysis(datapath)
    # input_data, ground_truth = generate_filename_list(datapath, file_key, dir_key)
    # input_data.sort()
    # ground_truth.sort()
    # print(input_data)
    # # data_preprocessing(datapath, file_key, dir_key)
    # train_dataset = ImageDataset(datapath, train='train', data_split=(0.7,0.3), crop_size=256)
    # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # data = next(iter(train_dataloader))
    # # print(data['gt'].shape)
    # for index, data in enumerate(train_dataloader):
    #     fig, (ax1, ax2) = plt.subplots(1,2)
    #     ax1.imshow(data['input'][0,0])
    #     ax2.imshow(data['gt'][0,0])
    #     plt.show()
    
  
  