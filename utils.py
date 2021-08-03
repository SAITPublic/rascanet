"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

import copy
import torch
from torch.autograd import Variable
import numpy as np


def print_model_ops_memory(model, iwidth, iheight, multiply_adds=False, device='cpu'):
    list_runtime_memory = []
    list_conv_2d = []
    list_conv_1d = []
    list_linear = []
    list_bn = []

    def bn_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        runtime_memory = (output_channels * output_height * output_width)
        list_runtime_memory.append(runtime_memory)

        flops = output_channels * output_height * output_width
        # list_bn.append(flops)
        list_bn.append(flops * batch_size)

    def conv2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        runtime_memory = (input_channels * input_height * input_width) + \
                         (output_channels * output_height * output_width)
        list_runtime_memory.append(runtime_memory)

        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width
        # list_conv_2d.append(flops)
        list_conv_2d.append(flops * batch_size)

    def conv1d_hook(self, input, output):
        batch_size, input_channels, input_size = input[0].size()
        output_channels, output_size = output[0].size()

        kernel_ops = self.kernel_size[0] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        runtime_memory = (input_channels * input_size) + (output_channels * output_size)
        list_runtime_memory.append(runtime_memory)

        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_size
        # list_conv_1d.append(flops)
        list_conv_1d.append(flops * batch_size)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        input_size = input[0].size(1)
        output_size = output.size(1)

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is None:
            bias_ops = 0
        else:
            bias_ops = self.bias.nelement()

        runtime_memory = input_size + output_size
        list_runtime_memory.append(runtime_memory)

        flops = (weight_ops + bias_ops)
        list_linear.append(flops * batch_size)

    def hook(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            if isinstance(net, torch.nn.Conv1d):
                net.register_forward_hook(conv1d_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            return
        for c in childrens:
            hook(c)

    model_clone = copy.deepcopy(model)
    hook(model_clone)

    input = Variable(torch.rand(3, iheight, iwidth).unsqueeze(0), requires_grad=False)
    if device == 'cuda':
        _ = model_clone(input.cuda())
    else:
        _ = model_clone(input)

    max_runtime_memory = max(list_runtime_memory)
    total_flops = sum(list_conv_2d) + sum(list_linear) + sum(list_conv_1d) + sum(list_bn)

    print('  + Input Memory: %.2fKB' % (iwidth * iheight * 3 / 1e3))
    print('  + Peak Memory: %.2fKB' % (max_runtime_memory / 1e3))
    print('  + Number of OPs: %.2fM' % (total_flops / 1e6))

    return total_flops / 1e6


def print_model_weight_size(model, verbose=False):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if verbose:
                if param.dim() > 1:
                    print(str(name) + ':' + 'x'.join(str(x) for x in list(param.size())) + '=' + str(num_param))
                else:
                    print(str(name) + ':' + str(num_param))
            total_param += num_param

    print('  + Kernel Memory: %.2fKB' % (total_param / 1e3))


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
