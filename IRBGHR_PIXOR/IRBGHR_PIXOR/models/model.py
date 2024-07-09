import torch
import torch.nn as nn

from IRBGHR_PIXOR.models.backbones.rpn import RPN
from IRBGHR_PIXOR.models.backbones.mobilepixor import MobilePixorBackBone
from IRBGHR_PIXOR.models.backbones.pixor import PixorBackBone
from IRBGHR_PIXOR.models.heads.cnn import Header

class CustomModel(nn.Module):
    def __init__(self, cfg, num_classes = 4):
        super(CustomModel, self).__init__()
        if cfg["backbone"] == "mobilepixor":
            self.backbone = MobilePixorBackBone()
        elif cfg["backbone"] == "pixor":
            self.backbone = PixorBackBone()
        elif cfg["backbone"] == "rpn":
            self.backbone = RPN()

        self.num_classes = num_classes
        if cfg["cls_encoding"] == "binary":
            self.num_classes += 1

        self.header = Header(self.num_classes, cfg["backbone_out_dim"])

    def forward(self, x):
        features = self.backbone(x)
        pred = self.header(features)

        return pred


if __name__ == "__main__":
    cfg = {
        "backbone": "mobilepixor"
    }

    model = CustomModel(cfg)
    for p in model.backbone.parameters():
        print(p)
    #print(model.backbone.parameters)