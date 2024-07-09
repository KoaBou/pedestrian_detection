import torch
from torch import nn

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Head(nn.Module):
    def __init__(self, in_channels, out):
        super(Head, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels)
        self.conv2 = conv3x3(in_channels, in_channels)
        self.head = nn.Conv2d(in_channels, out, kernel_size=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        head = self.head(x)
         
        return head

class Header(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(Header, self).__init__()

        self.cls = Head(in_channels, num_classes)
        self.offset = Head(in_channels, 2)
        self.size = Head(in_channels, 2)
        self.yaw = Head(in_channels, 2)


    def forward(self, x):
        cls = self.cls(x)
        offset = self.offset(x)
        size = self.size(x)
        yaw = self.yaw(x)

        pred = {"cls": cls, "offset": offset, "size": size, "yaw": yaw}  

        return pred
