#  Code Refer to: https://github.com/wolny/pytorch-3dunet
from functools import partial
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import layers
import collections

Conv_Bn_Activation = layers.Conv_Bn_Activation
DoubleConv = layers.DoubleConv
#  TODO: upsampling align_corner
# TODO: general solution

class Torchvision_backbone(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super(Torchvision_backbone, self).__init__()
        self.pretrained = pretrained
        if backbone == 'vgg16':
            self.model = models.vgg16(pretrained=self.pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=self.pretrained)
        elif backbone == 'resnext50':
            self.model = models.resnext50_32x4d(pretrained=self.pretrained)
        elif backbone == 'wide_resnet':
            self.model = models.wide_resnet50_2(pretrained=self.pretrained)
        else:
            raise ValueError('Undefined Backbone Name.')
       
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    # def __init__(self, input_channels, num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True, align_corner=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x 
        

class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


def get_activation(name, *args, **kwargs):
    if name == 'relu':
        return nn.ReLU(inplace=True, *args, **kwargs)


class ImageClassifier(nn.Module):
    def __init__(self, name, in_channels, out_channels, activation='relu', bsckbone='resnet50', pretrained=True):
        super(ImageClassifier, self).__init__()
        self.out_channels = out_channels
        self.encoder = Torchvision_backbone(bsckbone, pretrained=pretrained)
        # TODO: dynamic change node number
        self.fc1 = nn.Linear(1000, out_channels)
        self.activation_func = get_activation(activation)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.activation_func(x)
        return x


class UNet_2d(nn.Module):
    def __init__(self, input_channels, num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True):
        super(UNet_2d, self).__init__()
        self.bilinear = bilinear
        self.name = '2d_unet'

        # model2 = nn.Sequential(collections.OrderedDict([
        #         ('conv1', Conv_Bn_Activation(in_channels=root_channel, out_channels=root_channel)),
        #         ('conv2', nn.ReLU()),
        #         ('conv3', nn.Conv2d(20,64,5)),
        #         ('conv4', nn.ReLU())
        #         ]))

        self.conv1 = DoubleConv(in_channels=input_channels, out_channels=root_channel, mid_channels=root_channel)
        self.conv2 = DoubleConv(in_channels=root_channel, out_channels=root_channel*2)
        self.conv3 = DoubleConv(in_channels=root_channel*2, out_channels=root_channel*4)
        self.conv4 = DoubleConv(in_channels=root_channel*4, out_channels=root_channel*8)

        self.intermedia = DoubleConv(in_channels=root_channel*8, out_channels=root_channel*8)

        self.conv5 = DoubleConv(in_channels=root_channel*16, out_channels=root_channel*4)
        self.conv6 = DoubleConv(in_channels=root_channel*8, out_channels=root_channel*2)
        self.conv7 = DoubleConv(in_channels=root_channel*4, out_channels=root_channel)
        self.conv8 = DoubleConv(in_channels=root_channel*2, out_channels=root_channel)

        self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
        # self.upsampling = F.interpolate(x, size=size)
        self.classifier = Conv_Bn_Activation(in_channels=root_channel, out_channels=num_class, activation=None)

    def forward(self, x):
        low_level = []
        x = self.conv1(x)
        low_level.append(x)
        x = self.pooling(x)

        x = self.conv2(x)
        low_level.append(x)
        x = self.pooling(x)
        
        x = self.conv3(x)
        low_level.append(x)
        x = self.pooling(x)
        
        x = self.conv4(x)
        low_level.append(x)
        x = self.pooling(x)

        x = self.intermedia(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear')

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv5(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear')
        
        x = torch.cat([x, low_level.pop()],1)
        x = self.conv6(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear')

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv7(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear')

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv8(x)
        return nn.Sigmoid()(self.classifier(x))


def get_model(model_config):
    def _model_class(class_name):
        modules = ['model']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz

    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


if __name__ == "__main__":
    # print(Torchvision_backbone('resnet50'))
    
    print(ImageClassifier(name='test', num_class=3))
    # print(UNet_2d(num_class=2))
    # num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True