"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

import torch.nn as nn
from modules.activation import get_activation


class CNN(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, padding, stride,
                 dilation=1, groups=1, use_act=True, use_bn=True, bias=False):
        super(CNN, self).__init__()
        self.out_channels = out_planes
        # Conv
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if use_bn else None
        self.activation = get_activation('relu') if use_act else None

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x
