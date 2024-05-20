import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from lib.pvtv2 import pvt_v2_b2
from lib.decoder import LSSNet


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    
class PVT_LSSNet(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_LSSNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.1)
        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'E:/notes/SegCode/pretrained_pth/pvt/pvt_v2_b2.pth'
        # path = "/home/ww/shy/Seg/SegCode/pretrained_pth/pvt/pvt_v2_b2.pth"
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = LSSNet(channels=[512, 320, 128, 64])
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x, x4, x3, x2, x1)

        p1 = F.interpolate(x1_o, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(x2_o, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(x3_o, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(x4_o, scale_factor=4, mode='bilinear')

        return p1, p2, p3, p4

