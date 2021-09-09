
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
from dataset import input_preprocess
from dataset import dataset_utils
from utils import configuration

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


# def generate_augment_samples(path, aug_config, dataset_config, mode):
#     dataset = ImageDataset(dataset_config, mode)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     crop_size = aug_config.crop_size
#     ratio = 0.75
#     for idx, data in enumerate(dataloader):
#         print(f'Sample {idx}')
#         inputs, labels = data['input'].numpy(), data['gt'].numpy()
#         # inputs, labels = inputs.to(device), labels.to(device)
#         height, width = inputs.shape[2:]
#         input_list = [inputs[0,0]]
#         label_list = [labels[0,0]]

#         # flipping
#         if aug_config.flip:
#             flipped_image, flipped_label = [], []
#             for image, label in zip(input_list, label_list):
#                 flip_image, flip_label = preprocessing.random_flip(image, label, 1.0)
#                 flipped_image.append(flip_image)
#                 flipped_label.append(flip_label)
#             input_list.extend(flipped_image)    
#             label_list.extend(flipped_label)    

#         # scaling
#         # cropping
#         if aug_config.crop:
#             cropped_image, cropped_label = [], []
#             for image, label in zip(input_list, label_list):
#                 def crop5(image):
#                     crop_dict = {}
#                     crop_dict['left_top'] = image[:crop_size, :crop_size]

#                     if crop_size/width < ratio and crop_size/height < ratio:
#                         crop_dict['right_bot'] = image[(height-crop_size):, (width-crop_size):]

#                         offset_height = (height - crop_size) // 2
#                         offset_width = (width - crop_size) // 2
#                         crop_dict['center'] = image[offset_height:height-offset_height, offset_width:width-offset_width]

#                     if crop_size/width < ratio:
#                         crop_dict['right_top'] = image[:crop_size, (width-crop_size):]

#                     if crop_size/height < ratio:
#                         crop_dict['left_bot'] = image[(height-crop_size):, :crop_size]
#                     crop_dict = list(crop_dict.values())
#                     return crop_dict

#                 cropped_image.extend(crop5(image))
#                 cropped_label.extend(crop5(label))
                    

#                 # cropped_image.append(image[:height, :width])
#                 # cropped_image.append(image[:height, :width])
#                 # cropped_label.append(label[:height, :width])
#             input_list = cropped_image
#             label_list = cropped_label
#         j = 1
#         for image, label in zip(input_list, label_list):
#             cv2.imwrite(os.path.join(path, f'{dataset_config.dir_key}_{idx:04d}_aug{j}.png'), image*255)
#             cv2.imwrite(os.path.join(path, f'{dataset_config.dir_key}_{idx:04d}_mask_aug{j}.png'), label*255)
#             j += 1
#         # plt.imshow(label_list[-1], 'gray')
#         # plt.show()


def minmax_normalization(image):
    img_max = np.max(image)
    img_min = np.min(image)
    return (image-img_min) / (img_max-img_min)


def convert_value(image, value_pair=None):
    # TODO: one-to-one mapping, non-repeat
    # TODO: label pair convert
    for k in value_pair:
        image[image==k] = value_pair[k]
    return image


# def convert_value(image):
#     return np.int32(image//255)

# class AbstractDastaset(Dataset):
#     def __init__(self, config, mode):
#         assert (mode=='train' or mode=='test'), f'Unknown executing mode [{mode}].'
#         self.dataset_config = config.dataset.train if mode == 'train' else config.dataset.val
#         self.model_config = config.model
#         self.preprocess_config = self.dataset_config.preprocess_config
#         self.is_data_augmentation = self.dataset_config.is_data_augmentation
#         self.mode = mode
#         self.transform = transforms.Compose([transforms.ToTensor()])
#         self.load_func = None

#         self.check_dataset_split(config.dataset.data_split)
#         self.input_data_files, self.ground_truth_files = self.get_dataset_path(
#             dataset_name, dataset_split=config.dataset.data_split, mode=self.mode)

#         data_path = os.path.join(config.dataset.index_path, f'{mode}.txt')
#         self.input_data = dataset_utils.load_content_from_txt(data_path)
#         self.input_data.sort()
#         # TODO: gt not exist condition
#         self.ground_truth = [f.replace('.png', '_mask.png') for f in self.input_data] 
#         # self.ground_truth = [f.split('.png')[0]+'_mask.png' for f in self.input_data] 
#         print(f"{self.mode}  Samples: {len(self.input_data)}")

#     def __len__(self):
#         return len(self.input_data_files)

#     def __getitem__(self, idx):
#         input_data = self.load_func(self.input_data_files[idx])
#         input_data = self.transform(input_data)
#         ground_truth = self.load_func(self.ground_truth_files[idx])
#         ground_truth = self.transform(ground_truth)
#         return {'input': input_data, 'gt': ground_truth}

#     def check_dataset_split(self, data_split):
#         assert (isinstance(data_split, list) or isinstance(data_split, tuple))
#         assert data_split[0] + data_split[1] == 1

#     def get_dataset_path(self, dataset_name, dataset_split, mode):
#         data_path = os.path.join(config.dataset.index_path, f'{mode}.txt')


# TODO: Write in inherit form
# TODO: inference mode (no gt exist)
# TODO: dir_key to select benign or malignant
class ImageDataset(Dataset):
    def __init__(self, config, mode):
        # TODO: dynamic
        assert (mode=='train' or mode=='test'), f'Unknown executing mode [{mode}].'
        self.dataset_config = config.dataset.train if mode == 'train' else config.dataset.val
        self.model_config = config.model
        
        self.preprocess_config = self.dataset_config.preprocess_config
        self.is_data_augmentation = self.dataset_config.is_data_augmentation
        self.min_resize_value = self.preprocess_config.min_resize_value
        self.max_resize_value = self.preprocess_config.max_resize_value
        self.scale_factor_step_size = self.preprocess_config.scale_factor_step_size
        self.crop_size = self.preprocess_config.crop_size
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

        data_path = os.path.join(config.dataset.index_path, f'{mode}.txt')
        self.input_data = dataset_utils.load_content_from_txt(data_path)
        self.input_data.sort()
        # TODO: gt not exist condition
        self.ground_truth = [f.replace('.png', '_mask.png') for f in self.input_data] 
        # self.ground_truth = [f.split('.png')[0]+'_mask.png' for f in self.input_data] 
        print(f"{self.mode}  Samples: {len(self.input_data)}")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        # # TODO: assert for the situation that in_channels=1 but value different between channels
        # # Load images
        # self.original_image = cv2.imread(self.input_data[idx])[...,0:self.model_config.in_channels]
        # # TODO: dynamic value pair
        # self.original_label = convert_value(
        #     image=cv2.imread(self.ground_truth[idx])[...,0:self.model_config.in_channels], value_pair={255: 1})

        # # input_image, gt_image = preprocessing.resize_to_range(self.original_image, self.original_label,
        # #     min_size=self.min_resize_value, max_size=self.max_resize_value, factor=self.scale_factor_step_size)
        # # # Data preprocessing
        # output_strides = self.model_config.output_strides
        # input_image, gt_image = preprocessing.output_strides_align(self.original_image, output_strides, self.original_label)

        # if self.dataset_config.is_data_augmentation:
        #     preprocessor = preprocessing.DataPreprocessing(self.dataset_config['preprocess_config'])
        #     input_image, gt_image = preprocessor(input_image, gt_image)
            
        # # Transform to Torch tensor
        # input_image = self.transform(input_image)
        # gt_image = self.transform(gt_image)
        # return {'input': input_image, 'gt': gt_image}


        # ++++++
        # TODO: assert for the situation that in_channels=1 but value different between channels
        # Load images
        self.original_image = cv2.imread(self.input_data[idx])[...,0:self.model_config.in_channels]
        # TODO: dynamic value pair
        self.original_label = convert_value(
            image=cv2.imread(self.ground_truth[idx])[...,0:self.model_config.in_channels], value_pair={255: 1})

        input_image, gt_image = preprocessing.resize_to_range(self.original_image, self.original_label,
            min_size=self.min_resize_value, max_size=self.max_resize_value, factor=self.scale_factor_step_size)

        if self.is_data_augmentation:
            preprocessor = input_preprocess.DataPreprocessing(self.dataset_config['preprocess_config'])
            input_image, gt_image = preprocessor(input_image, gt_image)

        # Pad image and label to have dimensions >= [crop_height, crop_width]
        image_shape = input_image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + max(self.crop_size[0] - image_height, 0)
        target_width = image_width + max(self.crop_size[1] - image_width, 0)
        
        input_image = preprocessing.pad_to_bounding_box(
            input_image, 0, 0, target_height, target_width, pad_value=0)
        if gt_image is not None:
            gt_image = preprocessing.pad_to_bounding_box(
                gt_image, 0, 0, target_height, target_width, pad_value=0)

        if self.is_data_augmentation:
            input_image, gt_image = preprocessing.random_crop(input_image, gt_image, self.crop_size)
            input_image, gt_image = preprocessing.random_flip(input_image, gt_image, 
                flip_prob=self.preprocess_config.flip_prob, flip_mode=self.preprocess_config.flip_mode)

        # # Standardize
        # input_image = preprocessing.standardize(input_image)
        # if gt_image is not None:
        #     gt_image = preprocessing.standardize(gt_image)

        # Transform to Torch tensor
        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)
        return {'input': input_image, 'gt': gt_image}
        # -----

  
# class AudioDataset():
#     def __init__(self, config, mode):
        

#     def __len__(self):
#         return len(self.input_data)

#     def __getitem__(self, idx):
#         pass


class ClassificationImageDataset(ImageDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        # dataset_config, model_config = config.dataset, config.model
        self.ground_truth = []
        for f in self.input_data:
            if 'benign' in f:
                self.ground_truth.append(0)
            if 'malignant' in f:
                self.ground_truth.append(1)
            if 'normal' in f:
                self.ground_truth.append(2)

    def __getitem__(self, idx):
        # Load images
        # input_image = cv2.imread(self.input_data[idx])[...,0:1]
        self.original_image = cv2.imread(self.input_data[idx])
        self.original_label = self.ground_truth[idx]

        input_image, _ = preprocessing.resize_to_range(self.original_image, label=None,
            min_size=self.min_resize_value, max_size=self.max_resize_value, factor=self.scale_factor_step_size)

        # Data preprocessing
        if self.is_data_augmentation:
            preprocessor = input_preprocess.DataPreprocessing(self.preprocess_config)
            input_image, _ = preprocessor(input_image)
        
        # Pad image and label to have dimensions >= [crop_height, crop_width]
        image_shape = input_image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + max(self.crop_size[0] - image_height, 0)
        target_width = image_width + max(self.crop_size[1] - image_width, 0)
        
        input_image = preprocessing.pad_to_bounding_box(
            input_image, 0, 0, target_height, target_width, pad_value=0)

        if self.is_data_augmentation:
            input_image, _ = preprocessing.random_crop(input_image, label=None, crop_size=self.crop_size)
            input_image, _ = preprocessing.random_flip(input_image, label=None, 
                flip_prob=self.preprocess_config.flip_prob, flip_mode=self.preprocess_config.flip_mode)

        # Standardize
        # input_image = preprocessing.standardize(input_image)

        # Transform to Torch tensor
        input_image = self.transform(input_image)
        return {'input': input_image, 'gt': self.original_label}

    # def __getitem__(self, idx):
        # # Load images
        # # input_image = cv2.imread(self.input_data[idx])[...,0:1]
        # input_image = cv2.imread(self.input_data[idx])
        # gt = self.ground_truth[idx]

        # # Data preprocessing
        # output_strides = self.model_config.output_strides
        # input_image, _ = preprocessing.output_strides_align(input_image, output_strides)
        # if self.dataset_config.is_data_augmentation:
        #     preprocessor = input_preprocess.DataPreprocessing(self.dataset_config['preprocess_config'])
        #     input_image, _ = preprocessor(input_image)
        
        # # Standardize
        # # input_image = preprocessing.standardize(input_image)

        # # Transform to Torch tensor
        # input_image = self.transform(input_image)
        # return {'input': input_image, 'gt': gt}