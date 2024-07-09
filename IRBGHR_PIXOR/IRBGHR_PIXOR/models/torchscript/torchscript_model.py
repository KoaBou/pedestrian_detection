import torch
import torch.nn as nn

from backbones.rpn import RPN
from backbones.mobilepixor import MobilePixorBackBone
from backbones.pixor import PixorBackBone
from torchscript.heads import GaussianHead

class TorchscriptModel(nn.Module):
    def __init__(self, cfg, out_size_factor, num_classes = 4):
        super(TorchscriptModel, self).__init__()
        if cfg["backbone"] == "mobilepixor":
            self.backbone = MobilePixorBackBone()
        elif cfg["backbone"] == "pixor":
            self.backbone = PixorBackBone()
        elif cfg["backbone"] == "rpn":
            self.backbone = RPN()

        self.header = GaussianHead(cfg["backbone_out_dim"], out_size_factor, num_classes )

    def forward(self, x, x_min: float, y_min: float, x_res: float, y_res: float, score_threshold: float):
        features = self.backbone(x)
        pred = self.header(features, x_min, y_min, x_res, y_res, score_threshold)

        return pred


