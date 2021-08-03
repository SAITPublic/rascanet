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


class Hardswish(nn.Module):

    def __init__(self, th):
        super().__init__()
        self.th = th

    def forward(self, x):
        y = x
        y = y * (y + self.th) / (2 * self.th)
        y.masked_fill_(x <= -self.th, 0)
        y.masked_fill_(x >= self.th, x)

        return y


def get_activation(type):
    if type == 'relu':
        activation = nn.ReLU()
    elif type == 'hswish':
        activation = nn.Hardswish()
    elif type == 'tanh':
        activation = nn.Tanh()

    return activation
