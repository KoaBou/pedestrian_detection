import torch
from torch import nn
import torch.nn.functional as F

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

        return cls, offset, size, yaw


class GaussianHead(nn.Module):
    def __init__(self, backbone_out_dim, out_size_factor, num_classes = 4):
        super(GaussianHead, self).__init__()
        #self.header = Header(num_classes, backbone_out_dim)
        self.out_size_factor = out_size_factor
        self.cls = Head(backbone_out_dim, num_classes)
        self.offset = Head(backbone_out_dim, 2)
        self.size = Head(backbone_out_dim, 2)
        self.yaw = Head(backbone_out_dim, 2)

    def forward(self, features, x_min: float, y_min: float, x_res: float, y_res: float, score_threshold: float):
        #cls, offset, size, yaw = self.header(features)
        cls = self.cls(features)
        offset = self.offset(features)
        size = self.size(features)
        yaw = self.yaw(features)
        cls = cls.sigmoid()

        offset_pred = offset[0].detach()
        size_pred = size[0].detach()
        yaw_pred = yaw[0].detach()
        cls_pred = cls[0].detach()

        cos_t, sin_t = torch.chunk(yaw_pred, 2, dim = 0)
        dx, dy = torch.chunk(offset_pred, 2, dim = 0)
        log_w, log_l = torch.chunk(size_pred, 2, dim = 0)

        cls_probs, cls_ids = torch.max(cls_pred, dim = 0)

    
        pooled = F.max_pool2d(cls_probs.unsqueeze(0), 3, 1, 1).squeeze()
        selected_idxs = torch.logical_and(cls_probs == pooled, cls_probs > score_threshold)

        y = torch.arange(cls.shape[2])
        x = torch.arange(cls.shape[3])

        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xx = xx.to(offset_pred.device)
        yy = yy.to(offset_pred.device)

        center_y = dy + yy *  y_res * self.out_size_factor + y_min
        center_x = dx + xx *  x_res * self.out_size_factor + x_min
        center_x = center_x.squeeze()
        center_y = center_y.squeeze()
        l = torch.exp(log_l).squeeze()
        w = torch.exp(log_w).squeeze()
        yaw2 = torch.atan2(sin_t, cos_t).squeeze()
        yaw = yaw2 / 2

        boxes = torch.cat([cls_ids[selected_idxs].reshape(-1, 1), 
                        cls_probs[selected_idxs].reshape(-1, 1), 
                        center_x[selected_idxs].reshape(-1, 1), 
                        center_y[selected_idxs].reshape(-1, 1), 
                        l[selected_idxs].reshape(-1, 1), 
                        w[selected_idxs].reshape(-1, 1), 
                        yaw[selected_idxs].reshape(-1, 1)], dim = 1)

        return boxes

