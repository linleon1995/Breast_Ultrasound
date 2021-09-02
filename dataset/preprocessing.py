
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


# TODO: Understand the code
def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    label_layout_is_chw=False,
                    method=cv2.INTER_LINEAR):
    """Resizes image or label so their sides are within the provided range.

    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum size is equal to min_size
        without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.

    An integer in `range(factor)` is added to the computed sides so that the
    final dimensions are multiples of `factor` plus one.

    Args:
        image: A 3D tensor of shape [height, width, channels].
        label: (optional) A 3D tensor of shape [height, width, channels] (default)
        or [channels, height, width] when label_layout_is_chw = True.
        min_size: (scalar) desired size of the smaller image side.
        max_size: (scalar) maximum allowed size of the larger image side. Note
        that the output dimension is no larger than max_size and may be slightly
        smaller than min_size when factor is not None.
        factor: Make output size multiple of factor plus one.
        align_corners: If True, exactly align all 4 corners of input and output.
        label_layout_is_chw: If true, the label has shape [channel, height, width].
        We support this case because for some instance segmentation dataset, the
        instance segmentation is saved as [num_instances, height, width].
        scope: Optional name scope.
        method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

    Returns:
        A 3-D tensor of shape [new_height, new_width, channels], where the image
        has been resized (with the specified method) so that
        min(new_height, new_width) == ceil(min_size) or
        max(new_height, new_width) == ceil(max_size).

    Raises:
        ValueError: If the image is not a 3D tensor.
    """
    new_tensor_list = []
    min_size = float(min_size)
    if max_size is not None:
        max_size = float(max_size)
        # Modify the max_size to be a multiple of factor plus 1 and make sure the
        # max dimension after resizing is no larger than max_size.
        if factor is not None:
            max_size = (max_size + (factor - (max_size - 1) % factor) % factor
                        - factor)

    [orig_height, orig_width, _] = image.shape
    orig_height = float(orig_height)
    orig_width = float(orig_width)
    orig_min_size = min(orig_height, orig_width)

    # Calculate the larger of the possible sizes
    large_scale_factor = min_size / orig_min_size
    large_height = int(orig_height * large_scale_factor) + 1
    large_width = int(orig_width * large_scale_factor) + 1
    large_size = np.stack([large_width, large_height])
    
    new_size = large_size
    if max_size is not None:
        # Calculate the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_size = max(orig_height, orig_width)
        small_scale_factor = max_size / orig_max_size
        small_height = int(orig_height * small_scale_factor) + 1
        small_width = int(orig_width * small_scale_factor) + 1
        small_size = np.stack([small_width, small_height])
        new_size = small_size if float(np.max(large_size)) > max_size else large_size
        # new_size = np.cond(
        #     float(np.max(large_size)) > max_size,
        #     lambda: small_size,
        #     lambda: large_size)
    # Ensure that both output sides are multiples of factor plus one.
    if factor is not None:
        new_size += (factor - (new_size - 1) % factor) % factor
    image = cv2.resize(image, (new_size[0], new_size[1]), interpolation=method)
    if len(image.shape)==2: image = image[...,np.newaxis]
    new_tensor_list.append(image)
    # new_tensor_list.append(tf.image.resize_images(
    #     image, new_size, method=method, align_corners=align_corners))
    if label is not None:
        if label_layout_is_chw:
            # Input label has shape [channel, height, width].
            resized_label = np.expand_dims(label, 3)
            resized_label = cv2.resize(resized_label, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
            # resized_label = tf.image.resize_nearest_neighbor(
            #     resized_label, new_size, align_corners=align_corners)
            resized_label = np.squeeze(resized_label, 3)
        else:
            # Input label has shape [height, width, channel].
            resized_label = cv2.resize(label, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
            # resized_label = tf.image.resize_images(
            #     label, new_size, method=cv2.INTER_NEARST,
            #     align_corners=align_corners)
        if len(resized_label.shape)==2: resized_label = resized_label[...,np.newaxis]
        new_tensor_list.append(resized_label)
    else:
        new_tensor_list.append(None)
    return new_tensor_list


def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
        image: 3-D tensor with shape [height, width, channels]
        offset_height: Number of rows of zeros to add on top.
        offset_width: Number of columns of zeros to add on the left.
        target_height: Height of output image.
        target_width: Width of output image.
        pad_value: Value to pad the image tensor with.

    Returns:
        3-D tensor of shape [target_height, target_width, channels].

    Raises:
        ValueError: If the shape of image is incompatible with the offset_* or
        target_* arguments.
    """
    image_rank = len(image.shape)
    image_shape = np.shape(image)
    height, width = image_shape[0], image_shape[1]
    assert image_rank == 3, 'Wrong image tensor rank, expected 3'
    assert target_width >= width, 'target_width must be >= width'
    assert target_height >= height, 'target_height must be >= height'
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    assert (after_padding_width >= 0 and after_padding_height >= 0), \
        'target size not possible with the given target offsets'

    paddings = (after_padding_height, 0, after_padding_width, 0)
    padded = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_CONSTANT, value=pad_value)
    return padded


def standardize(m, mean=None, std=None, eps=1e-10, channelwise=False, *kwargs):
    if mean is None:
        if channelwise:
            # normalize per-channel
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            mean = np.mean(m, axis=axes, keepdims=True)
            std = np.std(m, axis=axes, keepdims=True)
        else:
            mean = np.mean(m)
            std = np.std(m)
    return (m - mean) / np.clip(std, a_min=eps, a_max=None)


# class Standardize:
#     """
#     Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
#     """

#     def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
#         if mean is not None or std is not None:
#             assert mean is not None and std is not None
#         self.mean = mean
#         self.std = std
#         self.eps = eps
#         self.channelwise = channelwise

#     def __call__(self, m):
#         if self.mean is not None:
#             mean, std = self.mean, self.std
#         else:
#             if self.channelwise:
#                 # normalize per-channel
#                 axes = list(range(m.ndim))
#                 # average across channels
#                 axes = tuple(axes[1:])
#                 mean = np.mean(m, axis=axes, keepdims=True)
#                 std = np.std(m, axis=axes, keepdims=True)
#             else:
#                 mean = np.mean(m)
#                 std = np.std(m)

#         return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


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
        # self.PadToSquare = preprocess_config['PadToSquare']
        # self.ScaleToSize = preprocess_config['ScaleToSize']
        self.ScaleLimitSize = preprocess_config['ScaleLimitSize']
        self.RandRotate = preprocess_config['RandRotate']
        self.GaussianBlur = preprocess_config['GaussianBlur']

        # self.padding_height = preprocess_config['padding_height']
        # self.padding_width = preprocess_config['padding_width']
        self.padding_value = preprocess_config['padding_value']
        self.flip_prob = preprocess_config['flip_prob']
        self.min_scale_factor = preprocess_config['min_scale_factor']
        self.max_scale_factor = preprocess_config['max_scale_factor']
        self.step_size = preprocess_config['step_size']
        assert (preprocess_config['resize_method'] == 'Bilinear' or preprocess_config['resize_method'] == 'Cubic')
        self.resize_method = cv2.INTER_LINEAR if preprocess_config['resize_method'] == 'Bilinear' else cv2.INTER_CUBIC
        self.crop_size = preprocess_config['crop_size']
        # self.scale_size = preprocess_config['scale_size']
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

        # if self.PadToSquare:
        #     # TODO: pad to square, params
        #     pad_size = max(Hs, Ws)
        #     self.padding_height, self.padding_width = pad_size, pad_size

        #     if Hs < self.padding_height:
        #         top = (self.padding_height - Hs)//2
        #         bottom = self.padding_height - top - Hs
        #     else:
        #         top, bottom = 0, 0
        #     if Ws < self.padding_width:
        #         left = (self.padding_width - Ws)//2
        #         right = self.padding_width - left - Ws
        #     else:
        #         left, right = 0, 0
        #     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_value)
        #     if label is not None:
        #         label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_value)
        #     Hs, Ws = image.shape[0], image.shape[1]
        #     if SHOW_PREPROCESSING: 
        #         show_data_information(image, label, 'pad to square')
        #         print('    padding', top, bottom, left, right)

        if self.RandFlip:
            image, label = rand_flip(image, label, flip_prob=self.flip_prob)
            if SHOW_PREPROCESSING: 
                show_data_information(image, label, 'random flip')

        # if self.ScaleToSize:
        #     Hs, Ws = self.scale_size[0], self.scale_size[1]
        #     image = cv2.resize(image, (Hs, Ws), interpolation=self.resize_method)
        #     if label is not None:
        #         label = cv2.resize(label, (Hs, Ws), interpolation=cv2.INTER_NEAREST)
        #     if SHOW_PREPROCESSING: 
        #         show_data_information(image, label, 'sacle to size')

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

        if len(image.shape)==2: image = image[...,np.newaxis]
        if label is not None:
            if len(label.shape)==2: label = label[...,np.newaxis]
        return (image, label)


    # def __call__(self, image, label=None):
    #     self.original_image = image
    #     self.original_label = label
    #     H = image.shape[0]
    #     W = image.shape[1]
    #     Hs, Ws = H, W
    #     image = np.squeeze(image)
    #     if label is not None:
    #         label = np.squeeze(label)
    #     # TODO: try decorator for conditional show
    #     if SHOW_PREPROCESSING: 
    #         show_data_information(image, label, method='origin')

    #     #  TODO: if immput is 190X400 will this distortion affect result?
        
    #     # if self.ScaleLimitSize:
    #     #     scale_ratio = (self.crop_size[0]+1) / min(H, W)
    #     #     if scale_ratio > 1.0:
    #     #         Hs, Ws = int(H*scale_ratio), int(W*scale_ratio)
    #     #         image = cv2.resize(image, (Ws,Hs), interpolation=self.resize_method)
    #     #         if label is not None:
    #     #             label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)
    #     #     if SHOW_PREPROCESSING:
    #     #         show_data_information(image, label, 'scale limit size')

    #     # if self.PadToSquare:
    #     #     # TODO: pad to square, params
    #     #     pad_size = max(Hs, Ws)
    #     #     self.padding_height, self.padding_width = pad_size, pad_size

    #     #     if Hs < self.padding_height:
    #     #         top = (self.padding_height - Hs)//2
    #     #         bottom = self.padding_height - top - Hs
    #     #     else:
    #     #         top, bottom = 0, 0
    #     #     if Ws < self.padding_width:
    #     #         left = (self.padding_width - Ws)//2
    #     #         right = self.padding_width - left - Ws
    #     #     else:
    #     #         left, right = 0, 0
    #     #     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_value)
    #     #     if label is not None:
    #     #         label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_value)
    #     #     Hs, Ws = image.shape[0], image.shape[1]
    #     #     if SHOW_PREPROCESSING: 
    #     #         show_data_information(image, label, 'pad to square')
    #     #         print('    padding', top, bottom, left, right)

    #     # if self.RandFlip:
    #     #     image, label = rand_flip(image, label, flip_prob=self.flip_prob)
    #     #     if SHOW_PREPROCESSING: 
    #     #         show_data_information(image, label, 'random flip')

    #     # if self.ScaleToSize:
    #     #     Hs, Ws = self.scale_size[0], self.scale_size[1]
    #     #     image = cv2.resize(image, (Hs, Ws), interpolation=self.resize_method)
    #     #     if label is not None:
    #     #         label = cv2.resize(label, (Hs, Ws), interpolation=cv2.INTER_NEAREST)
    #     #     if SHOW_PREPROCESSING: 
    #     #         show_data_information(image, label, 'sacle to size')

    #     if self.RandScale:
    #         scale = get_random_scale(self.min_scale_factor, self.max_scale_factor, self.step_size)
    #         # print(scale)
    #         Hs, Ws = int(scale*Hs), int(scale*Ws)
    #         image = cv2.resize(image, (Ws,Hs), interpolation=self.resize_method)
    #         if label is not None:
    #             label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)
    #         if SHOW_PREPROCESSING:
    #             show_data_information(image, label, 'random scale')

    #     if self.RandRotate:
    #         image, label = rand_rotate(image, label, min_angle=self.min_angle, max_angle=self.max_angle)
    #         if SHOW_PREPROCESSING:
    #             show_data_information(image, label, 'rotate')

    #     if self.GaussianBlur:
    #         image, label = gaussian_blur(image, label)
    #         if SHOW_PREPROCESSING:
    #             show_data_information(image, label, 'gaussian blur')

    #     # if self.RandCrop:
    #     #     Ws = np.random.randint(0, Ws - self.crop_size[1] + 1, 1)[0]
    #     #     Hs = np.random.randint(0, Hs - self.crop_size[0] + 1, 1)[0]
    #     #     image = image[Hs:Hs + self.crop_size[0], Ws:Ws + self.crop_size[0]]
    #     #     if label is not None:
    #     #         label = label[Hs:Hs + self.crop_size[1], Ws:Ws + self.crop_size[1]]
    #     #     if SHOW_PREPROCESSING: 
    #     #         show_data_information(image, label, 'random crop')

    #     if len(image.shape)==2: image = image[...,np.newaxis]
    #     if len(label.shape)==2: label = label[...,np.newaxis]
    #     return (image, label)