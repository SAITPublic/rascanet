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
from modules.attention import RasterAttention
from modules.cnn import CNN
from modules.rnn import GRU
from modules.vertical_split import VerticalSplit
import config


class RaScaNet(nn.Module):

    def __init__(self, n_class=2):
        super(RaScaNet, self).__init__()

        ############ Split Layer ###########
        self.split = config.get('split')
        self.split_layer = VerticalSplit(split=self.split)

        ############ Layer ###########
        self.cnn_channels = config.get('cnn_channels')
        self.rnn_channel = config.get('rnn_channel')
        self.multi_head = config.get('multi_head')
        self.rnn_width = config.get('rnn_width')
        for i in range(0, self.multi_head):
            setattr(self, 'raster_cnn_{}'.format(i), RasterCNN(cnn_channels=self.cnn_channels))
        self.raster_att = RasterAttention(multi_head=self.multi_head,
                                          cnn_channel=self.cnn_channels[-1],
                                          rnn_channel=self.rnn_channel,
                                          skip_spatial_attention=config.get('skip_spatial_attention'),
                                          skip_channel_attention=config.get('skip_channel_attention'))

        self.raster_rnn = RasterRNN(self.rnn_channel, self.rnn_channel)
        self.raster_confidence = RasterConfidence(self.rnn_channel)
        self.out_classifier = FinalClassifier(self.rnn_channel, config.get('fc_projection'), n_class)

    def forward(self, input):

        conf_labels = []
        conf_scores = []
        batch_size = input.size(0)
        hidden = torch.zeros((batch_size, self.rnn_channel, self.rnn_width), dtype=input.dtype, device=input.device)

        lines = self.split_layer(input)
        for line in lines:
            # cnn
            cnn_line_outs = []
            for i in range(0, self.multi_head):
                cnn_out = getattr(self, 'raster_cnn_{}'.format(i))(line).squeeze(2)
                cnn_line_outs.append(cnn_out)
            att = self.raster_att(cnn_line_outs, hidden)
            # rnn
            hidden = self.raster_rnn(att, hidden)
            conf_labels.append(self.out_classifier(hidden).detach().unsqueeze(1))
            conf_score = self.raster_confidence(hidden).unsqueeze(1)
            conf_scores.append(conf_score)
        # class
        out = self.out_classifier(hidden)

        return torch.cat(conf_labels, dim=1), torch.cat(conf_scores, dim=1), out

    def forward_fast(self, input, step=1):

        conf_labels = []
        conf_scores = []
        batch_size = input.size(0)
        hidden = torch.zeros((batch_size, self.rnn_channel, self.rnn_width), dtype=input.dtype, device=input.device)

        lines = self.split_layer(input, step)
        lines_cat = torch.cat(lines, dim=0)
        cnn_line_outs = []
        for i in range(0, self.multi_head):
            cnn_out = getattr(self, 'raster_cnn_{}'.format(i))(lines_cat).squeeze(2)
            sz = cnn_out.shape
            cnn_line_outs.append(torch.reshape(cnn_out, (len(lines), batch_size, sz[1], sz[2])))

        for idx in range(len(lines)):
            tmp_line_outs = []
            for i in range(0, self.multi_head):
                tmp_line_outs.append(cnn_line_outs[i][idx, :, :, :])
            att = self.raster_att(tmp_line_outs, hidden)

            hidden = self.raster_rnn(att, hidden)
            conf_labels.append(self.out_classifier(hidden).detach().unsqueeze(1))
            conf_score = self.raster_confidence(hidden)
            conf_scores.append(conf_score)

        # class
        out = self.out_classifier(hidden)

        return torch.cat(conf_labels, dim=1), torch.cat(conf_scores, dim=1), out

    def forward_early_terminate(self, input):

        batch_size = input.size(0)
        hidden = torch.zeros((batch_size, self.rnn_channel, self.rnn_width), dtype=input.dtype, device=input.device)

        lines = self.split_layer(input)
        cnt = 0
        skip = 0
        for t in range(len(lines)):
            if skip:
                skip -= 1
                continue
            cnt += 1
            line = lines[t]
            cnn_line_outs = []
            for i in range(0, self.multi_head):
                cnn_out = getattr(self, 'raster_cnn_{}'.format(i))(line).squeeze(2)
                cnn_line_outs.append(cnn_out)
            att = self.raster_att(cnn_line_outs, hidden)
            hidden = self.raster_rnn(att, hidden)
            out = self.out_classifier(hidden)
            if F.softmax(out, dim=-1)[:, 1] <= 0.005:
                skip = 1
            if F.softmax(out, dim=-1)[:, 1] >= 0.99:
                return out, cnt / len(lines)
        # class
        out = self.out_classifier(hidden)

        return out, cnt / len(lines)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class RasterCNN(nn.Module):

    def __init__(self, cnn_channels):
        super(RasterCNN, self).__init__()

        channel_0 = cnn_channels[0]
        channel_1 = cnn_channels[1]
        channel_2 = cnn_channels[2]
        channel_3 = cnn_channels[3]

        self.conv_block1 = CNN(in_planes=3, out_planes=channel_0, kernel_size=(3, 3), stride=1, padding=(0, 1))
        self.conv_block2 = CNN(in_planes=channel_0, out_planes=channel_1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_block3 = CNN(in_planes=channel_1, out_planes=channel_2, kernel_size=(3, 3), stride=1, padding=(0, 1))
        self.conv_block4 = CNN(in_planes=channel_2, out_planes=channel_3, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.pool_2d = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.pool_2d(out)

        out = self.conv_block2(out)
        out = self.pool_2d(out)

        out = self.conv_block3(out)
        out = self.pool_2d(out)

        out = self.conv_block4(out)
        out = self.pool_2d(out)

        return self.drop_out(out)


class RasterRNN(nn.Module):

    def __init__(self, in_channel, hidden_channel):
        super(RasterRNN, self).__init__()
        self.gru = GRU(in_channel, hidden_channel)

    def forward(self, input, hidden):
        hidden = self.gru(input, hidden)

        return hidden


class RasterConfidence(nn.Module):

    def __init__(self, fc_input):
        super(RasterConfidence, self).__init__()
        # self.drop_out = nn.Dropout(p=0.1)
        self.out_classifier = nn.Sequential(
            Flatten(),
            nn.Linear(fc_input, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.drop_out(x)
        x = x.mean(dim=-1) + x.max(dim=-1)[0]

        return self.out_classifier(x)


class FinalClassifier(nn.Module):

    def __init__(self, fc_input, fc_projection, n_class):
        super(FinalClassifier, self).__init__()

        self.drop_out = nn.Dropout(p=0.1)
        if fc_projection == 0:
            self.out_classifier = nn.Sequential(
                nn.Linear(fc_input, 2),
            )
        else:
            self.out_classifier = nn.Sequential(
                nn.Linear(fc_input, fc_projection),
                nn.BatchNorm1d(fc_projection),
                get_activation('hswish'),
                nn.Linear(fc_projection, n_class),
            )

    def forward(self, x):

        x = self.drop_out(x)
        x = x.mean(dim=-1) + x.max(dim=-1)[0]

        return self.out_classifier(x)
