"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

import math
import torch
from torch.nn import Module
import torch.nn as nn


class Cell(Module):

    def __init__(self, input_size, hidden_size):
        super(Cell, self).__init__()
        self.hidden_size = hidden_size
        self.conv_reset = nn.Conv1d(input_size + hidden_size, hidden_size, kernel_size=1)
        self.conv_update = nn.Conv1d(input_size + hidden_size, hidden_size, kernel_size=1)
        self.conv_h = nn.Conv1d(input_size + hidden_size, hidden_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        input_h = torch.cat([input, h], dim=1)
        reset_h = self.sigmoid(self.conv_reset(input_h))
        update_h = self.sigmoid(self.conv_update(input_h))

        input_new_h = torch.cat([input, h * reset_h], dim=1)
        new_h = self.activation(self.conv_h(input_new_h))

        h = (1 - update_h) * new_h + update_h * h

        return h


class GRU(Module):

    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = Cell(input_size, hidden_size)

    def forward(self, input, initial_state=None):
        if initial_state is None:
            zeros = torch.zeros(input.size(0),
                                self.hidden_size,
                                dtype=input.dtype,
                                device=input.device)
            initial_state = zeros

        states = initial_state
        output = self.cell(input, states)

        return output

    def __repr__(self):
        s = '{name}(input_size={input_size}, hidden_size={hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
