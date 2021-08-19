from numpy.lib.arraysetops import isin
from numpy.lib.type_check import _imag_dispatcher
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from model import UNet_2d
from torch.utils.data import Dataset, DataLoader
from dataset.dataloader import ImageDataset, data_analysis
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
    

# def save_file_names():
#     with open(rf'C:\Users\test\Desktop\Leon\Projects\Defect_Segmentation\s.txt', 'a+') as fw:
#         for i in range(1, 576):
#             fw.write(f'{i:04d}\n')

# def save_file_names():
#     with open(rf'C:\Users\test\Desktop\Leon\Projects\Defect_Segmentation\s.txt', 'a+') as fw:
#         for i in range(1, 576):
#             fw.write(f'{i:04d}\n')

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
            cv2.imwrite(os.path.join(save_path, save_name), mask)


def convert_mask(image):
    image = image // np.max(image) if np.max(image) > 0 else image
    return image


def convert_BU_mask(load_path, save_path):
    for idx, f in enumerate(os.listdir(load_path)):
        if f.split('.')[1] in ['png', 'PNG', 'jpg', 'jpeg']:
            print(f'step {idx+1}')
            mask = cv2.imread(os.path.join(load_path, f))
            mask = convert_mask(mask)
            # save_name = f.split('_label')[0] + f.split('_label')[1]
            cv2.imwrite(os.path.join(save_path, f), mask)


def BU_convert_BU_mask_main():
    load_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\masks'
    save_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\masks'
    convert_BU_mask(load_path, save_path)


def BU_load_content_from_txt():
    data_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\train.txt'
    content = load_content_from_txt(data_path)
    

def load_content_from_txt(path, access_mode='r'):
    with open(path, access_mode) as fw:
        # content = fw.readlines()
        content = fw.read().splitlines()
    return content


def string_filtering(s, filter):
    filtered_s = {}
    for f in filter:
        if f in s:
            filtered_s[f] = s
    if len(filtered_s) > 0:
        return filtered_s
    else:
        return None


def save_content_in_txt(content, path, access_mode='a+', dir=None):
    assert (isinstance(content, str) or 
            isinstance(content, list) or 
            isinstance(content, tuple) or 
            isinstance(content, dict))
    # TODO: overwrite warning
    with open(path, access_mode) as fw:
        def string_ops(s, dir, filter):
            pair = string_filtering(s, filter)
            return os.path.join(dir, list(pair.keys())[0], list(pair.values())[0])

        if isinstance(content, str):
            if dir is not None:
                content = string_ops(content, dir, filter=['benign', 'malignant', 'normal'])
                # content = os.path.join(dir, content)
            fw.write(content)
        else:
            for c in content:
                if dir is not None:
                    c = string_ops(c, dir, filter=['benign', 'malignant', 'normal'])
                    # c = os.path.join(dir, c)
                fw.write(f'{c}\n')
            
# TODO: General solution
def save_split_file_name_main():
    data_split = (0.7, 0.3)
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT'
    for dir in ['benign', 'malignant', 'normal']:
        input_data, ground_truth = dataloader.generate_filename_list(
            data_path, file_key='mask', dir_key=dir, only_filename=True)
        input_data.sort()
        ground_truth.sort()
        split = int(len(input_data)*data_split[0])
        train_input_data, _ = input_data[:split], ground_truth[:split]
        val_input_data, _ = input_data[split:], ground_truth[split:]
        # idx = 0
        # for tf in train_input_data:
        #     train_input_data[idx] = tf.replace('.png', '')
        #     idx += 1
        # idx = 0
        # for vf in val_input_data:
        #     val_input_data[idx] = vf.replace('.png', '')
        #     idx += 1

        # train_save_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\index\train_BU.txt'
        # val_save_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\index\val_BU.txt'
        train_save_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\train.txt'
        val_save_path = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\valid.txt'
        dir = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT'
        save_content_in_txt(train_input_data, train_save_path, access_mode="a+", dir=dir)
        save_content_in_txt(val_input_data, val_save_path, access_mode="a+", dir=dir)


def BU_modify_filename():
    data_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\masks'
    os.chdir(data_path)
    for f in os.listdir(data_path):
        os.rename(f, f.replace('_mask', ''))


def get_range_of_mask(segmentation, key_value, gap=10):
    # TODO: multi-class case
    index = np.where(segmentation==key_value)
    mask_range = []
    img_height, img_width = segmentation.shape[:2]
    
    if len(index[1]) > 0:
        x_min, x_max = np.min(index[1]), np.max(index[1])
        x_min = 0 if x_min-gap < 0 else x_min-gap
        x_max = img_width if x_max+gap > img_width else x_max+gap
        mask_range.extend([x_min, x_max])
    else:
        mask_range = None
    if len(index[0]) > 0:
        y_min, y_max = np.min(index[0]), np.max(index[0])
        y_min = 0 if y_min-gap < 0 else y_min-gap
        y_max = img_height if y_max+gap > img_height else y_max+gap
        mask_range.extend([y_min, y_max])
    else:
        mask_range = None
    return mask_range


def BU_segmentation_label_to_detection_label():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_original\benign'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_detection'

    os.chdir(data_path)
    for f in os.listdir(data_path):
        # Load segmentation
        mask = cv2.imread(f)
        img_height, img_width = mask.shape[:2]

        # Convert
        mask_range = get_range_of_mask(mask[...,2], key_value=255)

        # Save detection
        if mask_range is not None:
            x_min, x_max, y_min, y_max = mask_range
            mask_range = [0.0, ((x_min+x_max)/2)/img_width, ((y_min+y_max)/2)/img_height, (x_max-x_min)/img_width, (y_max-y_min)/img_height]
        
        draw_bbox_on_image(mask_range, mask)

def DAGM_draw_bbox_on_image():
    # TODO: Handle multiple targets
    # TODO: Handle multiple images
    img_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\YOLO\Release\ditTool\DataAugmentation\src_img\0577.PNG'
    bbox_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\YOLO\Release\ditTool\DataAugmentation\src_label\0577.txt'
    image = cv2.imread(img_path)
    with open(bbox_path, 'r') as fw:
        coord_str = fw.readline()
    if len(coord_str) > 0:
        coord_str = coord_str.split(' ')
        bbox = [float(c) for c in coord_str]
    else:
        bbox = None
    draw_bbox_on_image(bbox, image)
    

def draw_bbox_on_image(bbox, image):
    img_height, img_width = image.shape[:2]
    if bbox is not None:
        bbox_c, bbox_x, bbox_y, bbox_w, bbox_h = bbox
        start = (int(img_width*(bbox_x-bbox_w/2)), int(img_height*(bbox_y-bbox_h/2)))
        end = (int(img_width*(bbox_x+bbox_w/2)), int(img_height*(bbox_y+bbox_h/2)))
        cv2.rectangle(image, start, end, (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()


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
    # pred_and_label()
    save_split_file_name_main()
    # BU_load_content_from_txt()
    # BU_modify_filename()
    # BU_convert_BU_mask_main()
    # DAGM_draw_bbox_on_image()
    # BU_segmentation_label_to_detection_label()
    # image_and_label()
    # convert_DAGM_mask_main()

