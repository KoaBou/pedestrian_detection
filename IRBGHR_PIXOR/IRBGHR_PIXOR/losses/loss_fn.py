import torch
import torch.nn as nn
import torch.nn.functional as F

from focal_loss import modified_focal_loss, focal_loss
from l1_loss import l1_loss

class LossFunction(nn.Module):
    def __init__(self, cls_encoding):
        super(LossFunction, self).__init__()
        self.cls_encoding = cls_encoding

    def forward(self, pred, target):
        if self.cls_encoding == "binary":
            cls_loss = focal_loss(pred["cls"], target["cls"])
        else:
            cls_loss = modified_focal_loss(pred["cls"], target["cls"])
            #print("Run with modify_focal_loss")
        offset_loss = l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        loss = cls_loss + offset_loss + size_loss + yaw_loss

        loss_dict = {
            "loss": loss,
            "cls": cls_loss.item(),
            "offset": offset_loss.item(),
            "size": size_loss.item(),
            "yaw": yaw_loss.item()
        }

        return loss_dict