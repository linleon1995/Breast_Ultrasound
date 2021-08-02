import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
import collections

Conv_Bn_Activation = layers.Conv_Bn_Activation
DoubleConv = layers.DoubleConv
#  TODO: upsampling align_corner
# TODO: general solution
class UNet_2d(nn.Module):
    def __init__(self, input_channels, num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True):
        super(UNet_2d, self).__init__()
        self.bilinear = bilinear

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

# if __name__ == "__main__":
#     # print(Conv_Bn_Activation(32, 64))
#     print(UNet_2d(num_class=2))
#     # num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True