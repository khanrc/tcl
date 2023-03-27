# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch.nn as nn

from models.builder import MODELS
from models.tcl.modules import ResConv


@MODELS.register_module()
class GDecoder(nn.Module):
    def __init__(self, C, kernel_size, norm, act, double, n_layers=2, **kwargs):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv(
                    C, C,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    upsample=True,
                    norm=norm,
                    activ=act,
                    double=double,
                    gate=True
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
