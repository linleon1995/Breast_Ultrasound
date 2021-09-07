import cv2
import numpy as np
from dataset import preprocessing
scale_to_limit_size = preprocessing.scale_to_limit_size
rand_flip = preprocessing.rand_flip
rand_scale = preprocessing.rand_scale
rand_rotate = preprocessing.rand_rotate
gaussian_blur = preprocessing.gaussian_blur
random_crop = preprocessing.random_crop
show_data = preprocessing.show_data
SHOW_PREPROCESSING = True


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
        # self.ScaleLimitSize = preprocess_config['ScaleLimitSize']
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
        self.show_preprocess = preprocess_config['show_preprocess']

    # def __call__(self, image, label=None):
    #     self.original_image = image
    #     self.original_label = label
    #     H = image.shape[0]
    #     W = image.shape[1]
    #     # Hs, Ws = H, W
    #     image = np.squeeze(image)
    #     if label is not None:
    #         label = np.squeeze(label)
    #     if self.show_preprocess:
    #         print('method: original')
    #         show_data(image, label)

    #     #  TODO: if input is 190X400 will this distortion affect result?
    #     if self.ScaleLimitSize:
    #         image, label = scale_to_limit_size(image, label, crop_size=self.crop_size)

    #     if self.RandFlip:
    #         image, label = rand_flip(image, label, flip_prob=self.flip_prob)

    #     if self.RandScale:
    #         image, label = rand_scale(
    #             image, label, self.min_scale_factor, self.max_scale_factor, self.step_size, self.resize_method)

    #     if self.RandRotate:
    #         image, label = rand_rotate(image, label, min_angle=self.min_angle, max_angle=self.max_angle)

    #     if self.GaussianBlur:
    #         image, label = gaussian_blur(image, label)

    #     if self.RandCrop:
    #         image, label = random_crop(image, label, self.crop_size)

    #     if len(image.shape)==2: image = image[...,np.newaxis]
    #     if label is not None:
    #         if len(label.shape)==2: label = label[...,np.newaxis]
    #     return (image, label)


    def __call__(self, image, label=None):
        self.original_image = image
        self.original_label = label
        H = image.shape[0]
        W = image.shape[1]
        Hs, Ws = H, W
        image = np.squeeze(image)
        if label is not None:
            label = np.squeeze(label)

        if self.show_preprocess:
            print('method: original')
            show_data(image, label)

        #  TODO: if immput is 190X400 will this distortion affect result?
        
        # if self.ScaleLimitSize:
        #     scale_ratio = (self.crop_size[0]+1) / min(H, W)
        #     if scale_ratio > 1.0:
        #         Hs, Ws = int(H*scale_ratio), int(W*scale_ratio)
        #         image = cv2.resize(image, (Ws,Hs), interpolation=self.resize_method)
        #         if label is not None:
        #             label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)
        #     if SHOW_PREPROCESSING:
        #         show_data_information(image, label, 'scale limit size')

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

        # if self.RandFlip:
        #     image, label = rand_flip(image, label, flip_prob=self.flip_prob)
        #     if SHOW_PREPROCESSING: 
        #         show_data_information(image, label, 'random flip')

        # if self.ScaleToSize:
        #     Hs, Ws = self.scale_size[0], self.scale_size[1]
        #     image = cv2.resize(image, (Hs, Ws), interpolation=self.resize_method)
        #     if label is not None:
        #         label = cv2.resize(label, (Hs, Ws), interpolation=cv2.INTER_NEAREST)
        #     if SHOW_PREPROCESSING: 
        #         show_data_information(image, label, 'sacle to size')

        if self.RandScale:
            image, label = rand_scale(
                image, label, self.min_scale_factor, self.max_scale_factor, self.step_size, self.resize_method)

        if self.RandRotate:
            image, label = rand_rotate(image, label, min_angle=self.min_angle, max_angle=self.max_angle)

        if self.GaussianBlur:
            image, label = gaussian_blur(image, label)

        # if self.RandCrop:
        #     Ws = np.random.randint(0, Ws - self.crop_size[1] + 1, 1)[0]
        #     Hs = np.random.randint(0, Hs - self.crop_size[0] + 1, 1)[0]
        #     image = image[Hs:Hs + self.crop_size[0], Ws:Ws + self.crop_size[0]]
        #     if label is not None:
        #         label = label[Hs:Hs + self.crop_size[1], Ws:Ws + self.crop_size[1]]
        #     if SHOW_PREPROCESSING: 
        #         show_data_information(image, label, 'random crop')

        if len(image.shape)==2: image = image[...,np.newaxis]
        if label is not None:
            if len(label.shape)==2: label = label[...,np.newaxis]
        return (image, label)