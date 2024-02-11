
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.transform_layers import NormalizeLayer
from torch.nn.utils import spectral_norm
from torchvision import models

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class CLIP_R50(BaseModel):
    def __init__(self, num_classes=10, freezing_layer=166):
        import clip
        last_dim = 1024
        super(CLIP_R50, self).__init__(last_dim, num_classes)
        self.in_planes = 64
        self.last_dim = 1024
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        mu = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
        self.norm = lambda x: ( x - mu ) / std

        self.backbone, transform = clip.load("RN50", device=device)
        self.backbone = self.backbone.visual

        i = 0
        num = freezing_layer
        for param in self.backbone.parameters():
            if i<num:
                param.requires_grad = False
            i+=1
      
    def penultimate(self, x, all_features=False):
        x = self.norm(x)
        x = self.backbone(x)
        x = F.normalize(x, dim=-1).to(torch.float32)
        return x

def Clip_R50_Pretrain(num_classes, freezing_layer=166):
    return CLIP_R50(num_classes=num_classes, freezing_layer=freezing_layer)

