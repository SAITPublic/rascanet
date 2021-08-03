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


class VerticalSplit(nn.Module):

    def __init__(self, split=5):
        super(VerticalSplit, self).__init__()
        self.split = split

    def forward(self, input, step=1):
        bottom_remove = input.size(2) % self.split
        if bottom_remove != 0:
            input = input[:, :, :input.size(2) - bottom_remove, :]
            # input = nn.ConstantPad2d((0,0,0,bottom_padding), 0)(input)

        out = input.split(self.split, dim=2)

        # returns tuple(list)
        return out[::step]

    def __repr__(self):
        s = ('{name}(split={split})')
        return s.format(name=self.__class__.__name__, **self.__dict__)
