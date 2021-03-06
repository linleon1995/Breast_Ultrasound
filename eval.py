
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.functional import align_tensors
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset.dataloader import ImageDataset
from dataset.input_preprocess import DataPreprocessing
from utils import train_utils
from utils import metrics
from utils import configuration
from core.unet import unet_2d
UNet_2d = unet_2d.UNet_2d
EVAL_DIR_KEY = ''
# TODO: solve device problem, check behavoir while GPU using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_2dunet_seg_test_config.yml'

def eval():
    # dataset
    # config = cfg.dataset_config
    config = configuration.load_config(CONFIG_PATH)
    dataset_config = config['dataset']
    dataset_config['dir_key'] = EVAL_DIR_KEY
    test_dataset = ImageDataset(config, mode=config.eval.running_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    net = UNet_2d(input_channels=1, num_class=1)
    checkpoint = os.path.join(config.eval.restore_checkpoint_path, config.eval.checkpoint_name)
    state_key = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state_key['net'])
    net = net.to(device)
    net.eval()
    total_precision, total_recall, total_dsc, total_iou = [], [], [], []
    # total_tp, total_fp, total_fn = 0, 0, 0
    evaluator = metrics.SegmentationMetrics(num_class=config.model.out_channels)
    if len(test_dataloader) == 0:
        raise ValueError('No Data Exist. Please check the data path or data_plit.')
    # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
    for i, data in enumerate(test_dataloader):
        print('Sample: {}'.format(i+1))
        inputs, labels = data['input'], data['gt']
        inputs, labels = inputs.to(device), labels.to(device)
        logits = net(inputs)
        prediction = torch.round(logits)
        # TODO: 
        # +++
        inputs = inputs.cpu()[0].detach().numpy()
        inputs = np.swapaxes(inputs, 0, 1)
        inputs = np.swapaxes(inputs, 1, 2)
        prediction = prediction.cpu()[0].detach().numpy()
        # print(prediction.shape)
        prediction = np.swapaxes(prediction, 0, 1)
        prediction = np.swapaxes(prediction, 1, 2)
        original_height, orignal_width = test_dataset.original_image.shape[:2]
        # print(original_height, orignal_width)
        prediction = cv2.resize(prediction, (orignal_width,original_height), interpolation=cv2.INTER_NEAREST)
        labels = test_dataset.original_label
        # ---
        # print(prediction.shape)
        evals = evaluator(labels, prediction)
        tp, fp, fn = evaluator.tp, evaluator.fp, evaluator.fn
        # if (tp + fp + fn) != 0:
        total_dsc.append(evals['f1'])
        total_iou.append(evals['iou'])

        total_precision.append(evals['precision'])
        total_recall.append(evals['recall'])
        # total_tp += tp
        # total_fp += fp
        # total_fn += fn
        # TODO: original image size prediction
        image_code = i + 1
        image_code = test_dataset.input_data[i].split('\\')[-1]
        # def temporal_mask_process(mask):
        #     zero_label = np.zeros_like(mask)
        #     new_mask = np.concatenate(
        #         [zero_label[...,np.newaxis], zero_label[...,np.newaxis], 128*mask[...,np.newaxis]], axis=2)
        #     path = rf'C:\Users\test\Desktop\Software\SVN\Algorithm\deeplabv3+\data\myDataset\exp\segmentation_results_valid_20000\result_masks'
        #     helping_mask = cv2.imread(os.path.join(path, image_code))
        #     test_H, test_W = helping_mask.shape[:2]
        #     new_mask = new_mask[:test_H, :test_W]
        #     return new_mask
        # # np_label = labels.cpu()[0,0].detach().numpy()
        # x = temporal_mask_process(prediction.cpu()[0,0].detach().numpy())
        # print(np_label.shape)
        # TODO: if EVAL_DIR_KEY = ''
        if not os.path.exists(os.path.join(config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY)):
            os.makedirs(os.path.join(config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY))

        if config.eval.save_segmentation_result or config.eval.show_segmentation_result:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
            ax1.imshow(inputs, 'gray')
            ax2.imshow(labels, 'gray')
            ax3.imshow(prediction, 'gray')
        if config.eval.save_segmentation_result:
            # image_code = i + 1
            # image_code = test_dataset.input_data[i].split('\\')[-1]
            fig.savefig(os.path.join(
                config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY, f'{EVAL_DIR_KEY}_{config.eval.running_mode}_{image_code}.png'))
            plt.close(fig)
        if config.eval.save_prediciton_only:
            cv2.imwrite(
                os.path.join(config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY, f'{image_code}'), prediction)
        if config.eval.show_segmentation_result:
            plt.show()
    # print(tp, evaluator.total_tp, total_tp)
    # TODO: grid in plot
    # precision = metrics.precision(evaluator.total_tp, evaluator.total_fp).item() if evaluator.total_tp != 0 else 0
    # recall = metrics.recall(evaluator.total_tp, evaluator.total_fn).item() if evaluator.total_tp != 0 else 0
    # print(30*'-')
    # print(f'TP:{evaluator.total_tp} FP:{evaluator.total_fp} FN:{evaluator.total_fn} TN:{evaluator.total_tn}')
    # print(f'total precision: {precision:.4f}')
    # print(f'total recall: {recall:.4f}\n')

    # mean_precision = sum(total_precision)/len(total_precision)
    # mean_recall = sum(total_recall)/len(total_recall)
    # mean_dsc = sum(total_dsc)/len(total_dsc) if len(total_dsc) != 0 else 0
    # mean_iou = sum(total_iou)/len(total_iou) if len(total_iou) != 0 else 0
    # if sum(total_dsc) == 0:
    #     std_dsc = 0
    # else:
    #     std_dsc = [(dsc-mean_dsc)**2 for dsc in total_dsc]
    #     std_dsc = ((sum(std_dsc) / len(std_dsc))**0.5).item()
    # print(f'mean precision: {mean_precision:.4f}')
    # print(f'mean recall: {mean_recall:.4f}')
    # print(f'mean DSC: {mean_dsc:.4f}')
    # print(f'std DSC: {std_dsc:.4f}')
    # print(f'mean IoU: {mean_iou:.4f}')

    precision = metrics.precision(evaluator.total_tp, evaluator.total_fp)
    recall = metrics.recall(evaluator.total_tp, evaluator.total_fn)
    dsc = metrics.f1(evaluator.total_tp, evaluator.total_fp, evaluator.total_fn)
    iou = metrics.iou(evaluator.total_tp, evaluator.total_fp, evaluator.total_fn)

    accuracy = metrics.accuracy(np.sum(evaluator.total_tp), np.sum(evaluator.total_fp), np.sum(evaluator.total_fn))
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_dsc = np.mean(dsc)
    mean_iou = np.mean(iou)
    # mean_accuracy = np.mean(accuracy)
    print(30*'-')
    print(f'total precision: {mean_precision:.4f}')
    print(f'total recall: {mean_recall:.4f}\n')
    print(f'total accuracy: {accuracy:.4f}\n')
    print(f'DSC: {dsc}')
    print(f'mean DSC: {mean_dsc:.4f}')
    print(f'IoU: {iou}')
    print(f'mean IoU: {mean_iou:.4f}')


    # TODO: write to txt or excel

if __name__ == "__main__":
    eval()
    