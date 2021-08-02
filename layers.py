"""common deep learning layers for building unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: Complete resnetblock
# TODO: add layer name

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, batch_norm=True, activation='relu'):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.conv1 = Conv_Bn_Activation(in_channels=in_channels, 
                                        out_channels=mid_channels, 
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        batch_norm=batch_norm,
                                        activation=activation)
        self.conv2 = Conv_Bn_Activation(in_channels=mid_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        batch_norm=batch_norm,
                                        activation=activation)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batch_norm=True, activation='relu'):
        super().__init__()
        padding = (kernel_size - 1) // 2
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if batch_norm is True:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation == 'relu':
            modules.append(nn.ReLU(inplace=True))
        self.conv_bn_activation = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_bn_activation(x)


# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, basic_module=ConventionalConv, conv_kernel_size=3, padding=1, pooling='max', pool_kernel_size=2):
#         super().__init__()
#         assert pooling in ['max', 'avg']
#         if pooling == 'max':
#             self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
#         else:
#             self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)

#         self.basic_module = basic_module(in_channels, out_channels,
#                                          encoder=True,
#                                          kernel_size=conv_kernel_size,
#                                          order=conv_layer_order,
#                                          num_groups=num_groups,
#                                          padding=padding)
#     def forward(self, x):
#         if self.pooling is not None:
#             x = self.pooling(x)
#         x = self.basic_module(x)
#         return x


# if __name__ == "__main__":
#     # print(Conv_Bn_Activation(32, 64))
#     print(DoubleConv(32, 64))