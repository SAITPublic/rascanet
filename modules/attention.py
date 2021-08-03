"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.activation import get_activation
import math
import config


class RasterAttention(nn.Module):

    def __init__(self, multi_head, cnn_channel, rnn_channel, skip_spatial_attention=0, skip_channel_attention=0):
        super(RasterAttention, self).__init__()
        # Channel-Attention
        self.channel_prj = ChannelProjection(cnn_channel * multi_head,
                                             rnn_channel,
                                             config.get('attention_channel'),
                                             skip_channel_attention)
        # Spatial-Attention
        self.skip_spatial_attention = skip_spatial_attention
        if not self.skip_spatial_attention:
            for i in range(0, multi_head):
                setattr(self, 'spatial_attention_{}'.format(i),
                        SpatialAttention(cnn_channel, rnn_channel))

    def forward(self, input_list, hidden):
        # spatial attention
        if not self.skip_spatial_attention:
            enc_list = []
            for i, input in enumerate(input_list):
                enc = getattr(self, 'spatial_attention_{}'.format(i))(input, hidden)
                enc_list.append(enc)
            input = torch.cat(enc_list, dim=1)
        else:
            input = torch.cat(input_list, dim=1)
        ## channel attention
        out = self.channel_prj(input, hidden)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, cnn_channel, rnn_channel, att_near=2):
        super(SpatialAttention, self).__init__()
        # self.query_scale = 1.0
        self.query_scale = 1 / math.sqrt(rnn_channel)
        self.att_near = att_near
        self.dropout = nn.Dropout(p=0.1)
        self.key_conv = nn.Conv1d(cnn_channel, rnn_channel, kernel_size=1, bias=False)
        self.value_conv = nn.Conv1d(cnn_channel, cnn_channel, kernel_size=1, bias=False)

    def forward(self, input, hidden):
        ###########################################
        k = self.key_conv(input)
        q = hidden * self.query_scale
        v = self.value_conv(input)
        ###########################################
        ## query x key -> 15x15
        attn = torch.matmul(q.transpose(1, 2), k)
        att_mask = torch.ones_like(attn)
        att_mask = att_mask.tril(self.att_near).triu(-self.att_near)
        attn = F.softmax(attn * att_mask, dim=-1)
        value = torch.matmul(v, attn)

        return input + self.dropout(value)

    def __repr__(self):
        s = ('{name}(att_near={att_near})')
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ChannelProjection(nn.Module):

    def __init__(self, cnn_channel, rnn_channel, att_channel=8, skip_attention=0):
        super(ChannelProjection, self).__init__()

        self.att_channel = att_channel
        self.skip_attention = skip_attention
        # channel att layer
        if not self.skip_attention:
            self.channel_att = nn.Sequential(
                nn.Conv1d(cnn_channel + rnn_channel, att_channel, kernel_size=1),
                get_activation('relu'),
                nn.Conv1d(att_channel, cnn_channel, kernel_size=1),
                nn.Sigmoid(),
            )
        # projection layer
        self.projection = nn.Sequential(
            nn.Conv1d(cnn_channel, rnn_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(rnn_channel),
            get_activation('relu'),
            nn.Conv1d(rnn_channel, rnn_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(rnn_channel),
            get_activation('relu'),
            nn.Dropout(p=0.1),
        )

    def forward(self, input, hidden=None):

        if not self.skip_attention:
            x = torch.cat([input, hidden], dim=1)
            input = input * self.channel_att(x)

        return self.projection(input)

    def __repr__(self):
        s = ('{name}(att_channel={att_channel}, skip_attention={skip_attention})')
        return s.format(name=self.__class__.__name__, **self.__dict__)
