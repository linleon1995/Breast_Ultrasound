
# from dataset.dataloader import data_preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import cfg
# SHOW_PREPROCESSING = cfg.SHOW_PREPROCESSING
SHOW_PREPROCESSING = False



class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)

        
def output_strides_align(input_image, output_strides, gt_image=None):
    H = input_image.shape[0]
    W = input_image.shape[1]
    top, left = (output_strides-H%output_strides)//2, (output_strides-W%output_strides)//2
    bottom, right = (output_strides-H%output_strides)-top, (output_strides-W%output_strides)-left
    input_image = cv2.copyMakeBorder(input_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)
    if gt_image is not None:
        gt_image = cv2.copyMakeBorder(gt_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)
    return input_image, gt_image


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.
    Args:
        min_scale_factor: Minimum scale value.
        max_scale_factor: Maximum scale value.
        step_size: The step size from minimum to maximum value.
    Returns:
        A random scale value selected between minimum and maximum value.
    Raises:
        ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return np.random.uniform(min_scale_factor, max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = list(np.linspace(min_scale_factor, max_scale_factor, num_steps))
    return scale_factors[np.random.randint(0, len(scale_factors))]


def rand_crop(image, label=None, size=None):
    pass

def rand_rotate(image, label=None, min_angle=0, max_angle=0, center=None, scale=1.0, borderValue=0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    assert max_angle > min_angle
    if min_angle == max_angle:
        angle = min_angle
    else:
        angle = np.random.uniform(min_angle, max_angle)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (w, h), borderValue)
    if label is not None:
        label = cv2.warpAffine(label, M, (w, h), borderValue)
    return (image, label)

def gaussian_blur(image, label=None, kernel_size=(7,7)):
    assert (isinstance(kernel_size, tuple) or isinstance(kernel_size, list))
    image = cv2.GaussianBlur(image, kernel_size, 0)
    if label is not None:
        label = cv2.GaussianBlur(label, kernel_size, 0)
    return (image, label)

def rand_flip(image, label=None, flip_prob=0.5):
    randnum = np.random.uniform(0.0, 1.0)
    if flip_prob > randnum:
        image = cv2.flip(image, 1)
        if label is not None:
            label = cv2.flip(label, 1)
    return (image, label)

def show_data_information(image, label=None, method=None):
    if method is not None:
        print(f'method: {method}')
    print(f'    Image (value) max={np.max(image)} min={np.min(image)}  (shape) {image.shape}')
    print(f'    Label (value) max={np.max(image)} min={np.min(image)}  (shape) {image.shape}')
    _, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(np.uint8(image), 'gray')
    if label is not None:
        ax2.imshow(np.uint8(label), 'gray')
    plt.show()

def HU_to_pixelvalue():
    pass

def z_score_normalize(image):
    return np.uint8((image-np.mean(image)) / np.std(image))

# TODO: default params
# TODO: test image only, label only, image_label pair
# TODO: params check
# TODO: check deeplab code for generalization
# TODO: rectangle cropping
# TODO: rectangle resize?
class DataPreprocessing():
    def __init__(self, preprocess_config):
        self.RandFlip = preprocess_config['HorizontalFlip']
        self.RandCrop = preprocess_config['RandCrop']
        self.RandScale = preprocess_config['RandScale']
        self.PadToSquare = preprocess_config['PadToSquare']
        self.ScaleToSize = preprocess_config['ScaleToSize']
        self.ScaleLimitSize = preprocess_config['ScaleLimitSize']
        self.RandRotate = preprocess_config['RandRotate']
        self.GaussianBlur = preprocess_config['GaussianBlur']

        self.padding_height = preprocess_config['padding_height']
        self.padding_width = preprocess_config['padding_width']
        self.padding_value = preprocess_config['padding_value']
        self.flip_prob = preprocess_config['flip_prob']
        self.min_scale_factor = preprocess_config['min_scale_factor']
        self.max_scale_factor = preprocess_config['max_scale_factor']
        self.step_size = preprocess_config['step_size']
        # TODO: 
        self.resize_method = cv2.INTER_LINEAR if preprocess_config['resize_method'] == 'Bilinear' else None
        self.crop_size = preprocess_config['crop_size']
        self.scale_size = preprocess_config['scale_size']
        self.min_angle = preprocess_config['min_angle']
        self.max_angle = preprocess_config['max_angle']

    def __call__(self, image, label=None):
        self.original_image = image
        self.original_label = label
        H = image.shape[0]
        W = image.shape[1]
        Hs, Ws = H, W
        image = np.squeeze(image)
        if label is not None:
            label = np.squeeze(label)
        # TODO: try decorator for conditional show
        if SHOW_PREPROCESSING: 
            show_data_information(image, label, method='origin')

        #  TODO: if immput is 190X400 will this distortion affect result?
        
        if self.ScaleLimitSize:
            scale_ratio = (self.crop_size[0]+1) / min(H, W)
            if scale_ratio > 1.0:
                Hs, Ws = int(H*scale_ratio), int(W*scale_ratio)
                image = cv2.resize(image, (Ws,Hs), interpolation=self.resize_method)
                if label is not None:
                    label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)
            if SHOW_PREPROCESSING:
                show_data_information(image, label, 'scale limit size')

        if self.PadToSquare:
            # TODO: pad to square, params
            pad_size = max(Hs, Ws)
            self.padding_height, self.padding_width = pad_size, pad_size

            if Hs < self.padding_height:
                top = (self.padding_height - Hs)//2
                bottom = self.padding_height - top - Hs
            else:
                top, bottom = 0, 0
            if Ws < self.padding_width:
                left = (self.padding_width - Ws)//2
                right = self.padding_width - left - Ws
            else:
                left, right = 0, 0
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_value)
            Hs, Ws = image.shape[0], image.shape[1]
            if SHOW_PREPROCESSING: 
                show_data_information(image, label, 'pad to square')
                print('    padding', top, bottom, left, right)

        if self.RandFlip:
            image, label = rand_flip(image, label, flip_prob=self.flip_prob)
            if SHOW_PREPROCESSING: 
                show_data_information(image, label, 'random flip')

        if self.ScaleToSize:
            Hs, Ws = self.scale_size[0], self.scale_size[1]
            image = cv2.resize(image, (Hs, Ws), interpolation=self.resize_method)
            if label is not None:
                label = cv2.resize(label, (Hs, Ws), interpolation=cv2.INTER_NEAREST)
            if SHOW_PREPROCESSING: 
                show_data_information(image, label, 'sacle to size')

        if self.RandScale:
            scale = get_random_scale(self.min_scale_factor, self.max_scale_factor, self.step_size)
            # print(scale)
            Hs, Ws = int(scale*Hs), int(scale*Ws)
            image = cv2.resize(image, (Ws,Hs), interpolation=self.resize_method)
            if label is not None:
                label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)
            if SHOW_PREPROCESSING:
                show_data_information(image, label, 'random scale')

        if self.RandRotate:
            image, label = rand_rotate(image, label, min_angle=self.min_angle, max_angle=self.max_angle)
            if SHOW_PREPROCESSING:
                show_data_information(image, label, 'rotate')

        if self.GaussianBlur:
            image, label = gaussian_blur(image, label)
            if SHOW_PREPROCESSING:
                show_data_information(image, label, 'gaussian blur')

        if self.RandCrop:
            Ws = np.random.randint(0, Ws - self.crop_size[1] + 1, 1)[0]
            Hs = np.random.randint(0, Hs - self.crop_size[0] + 1, 1)[0]
            image = image[Hs:Hs + self.crop_size[0], Ws:Ws + self.crop_size[0]]
            if label is not None:
                label = label[Hs:Hs + self.crop_size[1], Ws:Ws + self.crop_size[1]]
            if SHOW_PREPROCESSING: 
                show_data_information(image, label, 'random crop')

        # image = np.expand_dims(image, 2)
        # if label is not None:
        #     label = np.expand_dims(label, 2)
        return (image, label)