from numpy.lib.arraysetops import isin
from numpy.lib.type_check import _imag_dispatcher
from sklearn import metrics
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import librosa.display
# from model import UNet_2d
# from model import ImageClassifier
from core.unet import unet_2d
from core.image_calssification import img_classifier
from torch.utils.data import Dataset, DataLoader
from dataset.dataloader import ImageDataset, ClassificationImageDataset, data_analysis
from dataset.input_preprocess import DataPreprocessing
import numpy as np
# from cfg import dataset_config
from dataset import dataloader
from utils import train_utils, metrics
import cv2
import os
from utils import configuration
from core import layers

UNet_2d = unet_2d.UNet_2d
ImageClassifier = img_classifier.ImageClassifier

CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_2dunet_cls_train_config.yml'

# def dataset_test():
#     train_dataset = ImageDataset(dataset_config, mode='train')
#     train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
#     for i, data in enumerate(train_dataloader):
#         fig, (ax1, ax2) = plt.subplots(1,2)
#         ax1.imshow(data['input'][0,0])
#         ax2.imshow(data['gt'][0,0])
#         plt.show()

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


def BU_detection_label_to_segmentation_label():
    data_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\YOLO\Release\yolo_main\v4\results'
    data_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\YOLO\Release\yolo_mark\data\template\label_loosely'
    data_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\YOLO\Release\yolo_mark\data\template\BU'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\yolo_BU_detection'
    img_format = 'png'
    os.chdir(data_path)
    data_list = [f for f in os.listdir(data_path) if 'txt' in f]
    data_list = ['benign (315).txt']
    for f in data_list:
        print(f)
        # Get the image size and create background
        img_name = f.replace('txt', img_format)
        img = cv2.imread(img_name)
        h, w = img.shape[:2]
        mask = np.zeros((h, w))

        # Get bounding box
        with open(f, 'r') as fw:
            detection = fw.readlines()
            for d in detection:
                num_detection = d.split(' ')[1:]
                num_detection = [float(c) for c in num_detection]
                bbox_x, bbox_y, bbox_w, bbox_h = num_detection

                # Produce mask
                start = (int(h*(bbox_y-bbox_h/2)), int(w*(bbox_x-bbox_w/2)))
                end = (int(h*(bbox_y+bbox_h/2)), int(w*(bbox_x+bbox_w/2))) 
                
                # mask[start[0]:end[0], start[1]:end[1]] = 1
                # print(mask.shape)
                # plt.imshow(img)
                # plt.imshow(mask, 'gray', alpha=0.5)
                # plt.show()
            
                _, ax = plt.subplots()
                ax.imshow(img)
                start = (int(w*(bbox_x-bbox_w/2)), int(h*(bbox_y-bbox_h/2)))
                rect = patches.Rectangle(start, w*bbox_w, h*bbox_h, linewidth=2, edgecolor='g', facecolor='none')
                # rect = patches.Rectangle((1,1), 100, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.show()
        # # Save mask
        # cv2.imwrite(os.path.join(save_path, img_name.replace('.jpg', '_bbox_mask.png')), mask)

def BU_segmentation_label_to_detection_label():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_original\malignant'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_detection\malignant'

    os.chdir(data_path)
    for f in os.listdir(data_path):
        # Load segmentation
        mask = cv2.imread(f)
        img_height, img_width = mask.shape[:2]
        print(f)
        # Convert
        mask_range = get_range_of_mask(mask[...,2], key_value=255)

        # Save detection
        if mask_range is not None:
            x_min, x_max, y_min, y_max = mask_range
            # convert to [center_x, center_y, ]
            mask_range = [((x_min+x_max)/2)/img_width, ((y_min+y_max)/2)/img_height, 
                           (x_max-x_min)/img_width, (y_max-y_min)/img_height]
        def save_detection_label():
            filename = f.replace('_mask.png', '')+'.txt'
            with open(os.path.join(save_path, filename), 'w+') as fw:
                if mask_range is not None:
                    fw.write('1 ')
                    for r in mask_range:
                        fw.write(f'{r:.6f} ')
        save_detection_label()

    # Redeaw bounding-box on image
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
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        start = (int(img_width*(bbox_x-bbox_w/2)), int(img_height*(bbox_y-bbox_h/2)))
        end = (int(img_width*(bbox_x+bbox_w/2)), int(img_height*(bbox_y+bbox_h/2)))
        cv2.rectangle(image, start, end, (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()


def BU_cls_dataloader_main():
    config = configuration.load_config(CONFIG_PATH)
    train_dataset = ClassificationImageDataset(config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=config.dataset.shuffle)
    for i, data in enumerate(train_dataloader):
        inputs, labels = data['input'], data['gt']
        y = F.one_hot(labels, num_classes=3)
        target = torch.FloatTensor(10).uniform_(0, 120).long()
        print(target.size())
        print(labels[0], y[0])
        print(inputs.size(), y.size())
        plt.imshow(inputs[0,0], 'gray')
        plt.show()


def test_new_eval_main():
    evaluator = metrics.SegmentationMetrics(['precision', 'recall'])
    target = np.array([[0,1,2],[0,1,2]])
    pred = np.array([[0,1,1],[0,1,2]])
    target = np.reshape(target, [-1])
    pred = np.reshape(pred, [-1])
    # print(target.shape, )
    evals = evaluator(label=target, pred=pred)
    print(evals)
    

# TODO: filtering mode for keyword, remove or add
def save_aLL_files_name(path, keyword=None):
    files = os.listdir(path)
    files.sort()
    with open(os.path.join(path, 'file_names.txt'), 'w+') as fw:
        for f in files:
            if keyword is not None:
                if keyword not in f:
                    fullpath = os.path.join(path, f)
                    fw.write(fullpath)    
                    fw.write('\n')
            else:
                fullpath = os.path.join(path, f)
                fw.write(fullpath)    
                fw.write('\n')

            
def BU_save_name_main():
    path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\YOLO\DataSet\BU\JPEGImages'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT\benign'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU\val\Images'
    save_aLL_files_name(path, keyword='mask')


def image_read_and_show(path, show=True):
    image = cv2.imread(path)
    if show:
        plt.imshow(image*128)
        plt.show()
    return image


def BU_image_read_and_show():
    path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\Kaggle_BU_convert_mask\benign\benign (100)_bbox_mask.png'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\yolo_BU_detection\pred\benign (264)_bbox_mask.jpg'
    path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_200000\result_masks\benign (274)_bbox_mask.png'
    image = image_read_and_show(path)
    print(image.shape)


def test_model_main():
    net = model.UNet_2d_backbone(in_channels=3, out_channels=3, basic_module=layers.DoubleConv, f_maps=32)
    print(net)


def dsc_for_deeplab_main():
    pred_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_200000\seg'
    target_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_20000\input_image\val'
    evaluator = metrics.SegmentationMetrics(num_class=2)
    mean_precision, mean_recall = [], []
    for l, p in zip(os.listdir(target_path), os.listdir(pred_path)):
        print(p)
        label = cv2.imread(os.path.join(target_path, l))[...,2]
        prediction = cv2.imread(os.path.join(pred_path, p))[...,2]
        prediction = prediction//128
        evals = evaluator(label, prediction)
        # mean_precision.append(evals['precision'])
        # mean_recall.append(evals['recall'])
    precision = metrics.precision(evaluator.total_tp, evaluator.total_fp)
    recall = metrics.recall(evaluator.total_tp, evaluator.total_fn)
    dsc = metrics.f1(evaluator.total_tp, evaluator.total_fp, evaluator.total_fn)
    iou = metrics.iou(evaluator.total_tp, evaluator.total_fp, evaluator.total_fn)

    accuracy = metrics.accuracy(np.sum(evaluator.total_tp), np.sum(evaluator.total_fp), np.sum(evaluator.total_fn))
    total_precision = np.mean(precision)
    total_recall = np.mean(recall)
    mean_dsc = np.mean(dsc)
    mean_iou = np.mean(iou)

    print(30*'-')
    print(f'total precision: {total_precision:.4f}')
    print(f'total recall: {total_recall:.4f}')
    print(f'total accuracy: {accuracy:.4f}\n')
    
    # mean_precision = sum(mean_precision) / len(mean_precision)
    # mean_recall = sum(mean_recall) / len(mean_recall)
    # print(f'mean precision: {mean_precision:.4f}')
    # print(f'mean recall: {mean_recall:.4f}\n')
    
    print(f'DSC: {dsc}')
    print(f'mean DSC: {mean_dsc:.4f}')
    print(f'IoU: {iou}')
    print(f'mean IoU: {mean_iou:.4f}')


    # for b in ['resnet18', 'resnet50', 'resnext50', 'wide_resnet']:
    #     for in_c in [1, 3]:
    #         for p in [True, False]:
    #             net = ImageClassifier(
    #                 backbone=b, in_channels=in_c, activation='sigmoid',
    #                 out_channels=3, pretrained=p, dim=1, output_structure=[100, 100])
    #             print(net)

def convert_value():
    path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_200000\result_masks'
    save_path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\BU_ckpt\detection'
    data_format = 'png'
    label_value = np.array([0,0,128])
    for f in os.listdir(path):
        if data_format in f:
            print(f)
            img = image_read_and_show(os.path.join(path, f), show=False)
            img = label_value * img
            # plt.imshow(img)
            # plt.show()
            cv2.imwrite(os.path.join(save_path, f), img)


def Timm_main():
    import timm
    from pprint import pprint
    model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    m = timm.create_model('efficientnet_b0', pretrained=True)
    print(m)
    

def contrast_enhance_main():
    #-----Reading the image-----------------------------------------------------
    img = cv2.imread(rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT\benign\benign (416).png', 1)
    plt.imshow(img, 'gray')
    plt.show()

    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    plt.imshow(lab)
    plt.show()

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    plt.imshow(l)
    plt.show()    
    plt.imshow(a)
    plt.show()
    plt.imshow(b)
    plt.show()
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    plt.imshow(cl)
    plt.show()
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    plt.imshow(limg, 'gray')
    plt.show()
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    plt.imshow(final, 'gray')
    plt.show()
    #_____END_____#


def torchaudio_augmentation():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset\0\0_0.wav'
    print(filename)
    waveform, sample_rate = torchaudio.load(filename)
    spectrogram = torchaudio.transforms.MelSpectrogram()
    spec = spectrogram(waveform)
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    plt.imshow(fbank)
    plt.show()
    
    # x , sr = librosa.load(filename)
    # plt.figure(figsize=(14, 5))
    # librosa.display.waveplot(x, sr)

if __name__ == "__main__":
    # test_new_eval_main()
    # augmentation_test()
    # precision_test()
    # pred_and_label()
    # save_split_file_name_main()
    # BU_load_content_from_txt()
    # BU_modify_filename()
    # BU_convert_BU_mask_main()
    # DAGM_draw_bbox_on_image()
    # BU_segmentation_label_to_detection_label()
    # image_and_label()
    # convert_DAGM_mask_main()
    # BU_cls_dataloader_main()
    # BU_save_name_main()
    BU_detection_label_to_segmentation_label()
    # test_model_main()
    # BU_image_read_and_show()
    # dsc_for_deeplab_main()
    # Timm_main()
    # contrast_enhance_main()
    # convert_value()
    # torchaudio_augmentation()
    pass