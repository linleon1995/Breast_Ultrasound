#  Code Refer to: https://github.com/wolny/pytorch-3dunet
from functools import partial
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import layers
import collections
import logging
from utils import train_utils

Conv_Bn_Activation = layers.Conv_Bn_Activation
_DoubleConv = layers._DoubleConv
#  TODO: upsampling align_corner
# TODO: general solution




def get_activation(name, *args, **kwargs):
    if name == 'relu':
        return nn.ReLU(inplace=True, *args, **kwargs)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(*args, **kwargs)

class PytorchResnetBuilder(nn.Module):
    def __init__(self, in_channels, backbone, pretrained=True, final_flatten=False):
        super(PytorchResnetBuilder, self).__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.pretrained = pretrained
        self.final_flatten = final_flatten
        self.model = self.get_model()

    def select_backbone(self):
        if self.backbone == 'resnet18':
            return models.resnet18(pretrained=self.pretrained)
        elif self.backbone == 'resnet34':
            return models.resnet34(pretrained=self.pretrained)
        elif self.backbone == 'resnet50':
            return models.resnet50(pretrained=self.pretrained)
        elif self.backbone == 'resnet101':
            return models.resnet101(pretrained=self.pretrained)
        elif self.backbone == 'resnet152':
            return models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError('Undefined Backbone Name.')

    def get_model(self):
        # Select backbone
        model  = self.select_backbone()

        # Decide using final classification part (GAP + fc)
        if not self.final_flatten:
            model = nn.Sequential(*list(model.children())[:-2])

        # Modify first convolution layer to accept different input channels
        if self.in_channels != 3:
            conv1 = model[0]
            conv1_out_c = conv1.out_channels
            conv1_ks = conv1.kernel_size
            conv1_stride = conv1.stride
            conv1_padding = conv1.padding
            model[0] = torch.nn.Conv1d(
                self.in_channels, conv1_out_c, conv1_ks, conv1_stride, conv1_padding, bias=False)
        return model

    def forward(self, x):
        return self.model(x)


class PytorchResnextBuilder(nn.Module):
    def __init__(self, in_channels, backbone, pretrained=True, final_flatten=False):
        super(PytorchResnextBuilder, self).__init__(in_channels, backbone, pretrained, final_flatten)

    def select_backbone(self):
        if self.backbone == 'resnext50':
            return models.resnext50_32x4d(pretrained=self.pretrained)
        elif self.backbone == 'resnext101':
            return models.resnext101_32x8d(pretrained=self.pretrained)
        else:
            raise ValueError('Undefined Backbone Name.')


class MultiLayerPerceptron(nn.Module):
    def __init__(self, structure, activation=None, out_activation=None, *args, **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        self.mlp = torch.nn.Sequential()
        self.activation = activation
        self.out_activation = out_activation
        assert isinstance(structure, (list, tuple)), 'Model structure "structure" should be list or tuple'
        assert len(structure) > 1, 'The length of structure should be at least 2 to define linear layer'

        for idx in range(len(structure)-1):
            in_channels, out_channels = structure[idx], structure[idx+1]
            self.mlp.add_module(f"fc{idx+1}", torch.nn.Linear(in_channels, out_channels))
            if self.activation is not None and idx+1 < len(structure)-1:
                self.mlp.add_module(f"{self.activation}{idx+1}", get_activation(self.activation))

        if self.out_activation is not None:
            out_dix = idx + 2 if self.out_activation == self.activation else 1
            self.mlp.add_module(f"{self.out_activation}{out_dix}", get_activation(self.out_activation))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


def creat_torchvision_backbone(in_channels, backbone, pretrained, final_flatten=True):
    if 'resnet' in backbone:
        return PytorchResnetBuilder(in_channels, backbone, pretrained, final_flatten)
    elif 'resnext' in backbone:
        return PytorchResnextBuilder(in_channels, backbone, pretrained, final_flatten)


def create_encoder(in_channels, backbone, pretrained=True, final_flatten=False):
    base_model = creat_torchvision_backbone(in_channels, backbone, pretrained=pretrained, final_flatten=False)
    base_layers = list(base_model.children())[0]
    conv1 = nn.Sequential(*base_layers[:3])
    conv2 = nn.Sequential(*base_layers[3:5])
    conv3 = nn.Sequential(*base_layers[5])
    conv4 = nn.Sequential(*base_layers[6])
    conv5 = nn.Sequential(*base_layers[7])
    encoders = [conv1, conv2, conv3, conv4, conv5]
    return nn.ModuleList(encoders)


class ImageClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, output_structure=None, activation=None, backbone='resnet50', 
                 pretrained=True, *args, **kwargs):
        super(ImageClassifier, self).__init__()
        self.out_channels = out_channels
        self.output_structure = output_structure
        if in_channels != 3 and pretrained:
            logging.info('Reinitialized first layer')

        self.encoder = creat_torchvision_backbone(in_channels, backbone, pretrained, final_flatten=True)

        # TODO: Define output sturcture outside this class
        module = list(self.encoder.children())[0]
        encoder_out_node = list(module.children())[-1].out_features

        if output_structure is not None:
            output_structure = [encoder_out_node] + output_structure + [out_channels]
            self.mlp = MultiLayerPerceptron(output_structure, 'relu', out_activation=activation)
        else:
            self.mlp = MultiLayerPerceptron([encoder_out_node, out_channels], 'relu', out_activation=activation)

        # self.fc1 = nn.Linear(encoder_out_node, out_channels)
        # self.activation_func = get_activation(activation, *args, **kwargs) if activation is not None else None
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        # x = self.fc1(x)
        # if self.activation_func is not None:
        #     x = self.activation_func(x)
        return x

class UNet_2d_backbone(nn.Module):
    def __init__(self, in_channels, out_channels, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, backbone='resnet50', pretrained=True, **kwargs):
        super(UNet_2d_backbone, self).__init__()
        self.out_channels = out_channels
        if in_channels != 3 and pretrained:
            logging.info('Reinitialized first layer')

        if isinstance(f_maps, int):
            f_maps = train_utils.number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoder(in_channels, backbone, pretrained=pretrained, final_flatten=False)
        
        # create decoder path
        self.decoder = layers.create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, 
                                              num_groups, upsample=True)

        self.classifier = Conv_Bn_Activation(in_channels=f_maps[0], out_channels=self.out_channels, activation=None)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if self.testing and self.final_activation is not None:
        #     x = self.final_activation(x)
        # return x
        x = self.classifier(x)
        return nn.Sigmoid()(self.classifier(x))



    # def forward(self, x):
    #     f1 = self.conv1(x)
    #     f2 = self.conv2(f1)
    #     f3 = self.conv3(f2)
    #     f4 = self.conv4(f3)
    #     f5 = self.conv5(f4)

    #     features = [f1, f2, f3, f4, f5]
    #     x = self.decoder(features)
    #     pass
    #     return x


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

        self.conv1 = _DoubleConv(in_channels=input_channels, out_channels=root_channel, mid_channels=root_channel)
        self.conv2 = _DoubleConv(in_channels=root_channel, out_channels=root_channel*2)
        self.conv3 = _DoubleConv(in_channels=root_channel*2, out_channels=root_channel*4)
        self.conv4 = _DoubleConv(in_channels=root_channel*4, out_channels=root_channel*8)

        self.intermedia = _DoubleConv(in_channels=root_channel*8, out_channels=root_channel*8)

        self.conv5 = _DoubleConv(in_channels=root_channel*16, out_channels=root_channel*4)
        self.conv6 = _DoubleConv(in_channels=root_channel*8, out_channels=root_channel*2)
        self.conv7 = _DoubleConv(in_channels=root_channel*4, out_channels=root_channel)
        self.conv8 = _DoubleConv(in_channels=root_channel*2, out_channels=root_channel)

        self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
        # self.upsampling = F.interpolate(x, size=size)
        self.classifier = Conv_Bn_Activation(in_channels=root_channel, out_channels=num_class, activation=None)

    def forward(self, x):
        # TODO: dynamic
        align_corner = False
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
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv5(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)
        
        x = torch.cat([x, low_level.pop()],1)
        x = self.conv6(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv7(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)

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
    # print(PytorchResnetBuilder('resnet50'))
    net = ImageClassifier(
        backbone='resnext50', in_channels=3, activation=None,
        out_channels=3, pretrained=True, dim=1)
    print(net)
    # print(UNet_2d(num_class=2))
    # num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True