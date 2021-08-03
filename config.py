"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

import os


def init(args):
    Config.instance().set("idx", 0)
    Config.instance().set("epoch", 0)
    Config.instance().set('train_class_acc', 0.0)
    Config.instance().set('test_class_acc', 0.0)

    if args.dataset == 'vww':
        # dataset_path should be db/VWW/coco2014/
        args.train_path = os.path.join(args.dataset_path, 'vww_annotations/instances_train.json')
        args.test_path = os.path.join(args.dataset_path, 'vww_annotations/instances_val.json')
        #
        if args.rsz_w == 240:
            args.multi_head = 2
            args.rsz_h = 210
            args.rnn_width = 15
            args.rnn_channel = 48
            args.fc_projection = 12
            args.cnn_channels = '6,11,21,42'
        elif args.rsz_w == 120:
            args.multi_head = 2
            args.rsz_h = 105
            args.rnn_width = 7
            args.rnn_channel = 40
            args.fc_projection = 10
            args.cnn_channels = '5,9,18,32'
        else:
            print('rsz_w currently not supported')
            exit()

    elif args.dataset == 'voc':
        # dataset_path should be db/PascalVOC/
        args.train_path = os.path.join(args.dataset_path, 'VOC_Train/')
        args.test_path = os.path.join(args.dataset_path, 'VOC_Test/')

    args.cnn_channels = [int(x) for x in args.cnn_channels.split(',')]
    args.attention_channel = int(args.rnn_channel / 4)
    set_all(args)

    return args


def set_all(args):
    for key, val in vars(args).items():
        set(key, val)


def get_index():
    idx = Config.instance().get('idx')
    Config.instance().set('idx', idx + 1)
    return idx


def get(key):
    return Config.instance().get(key)


def set(key, val):
    Config.instance().set(key, val)


def set_dict(key, val):
    Config.instance().set(key, val)


def update_acc(key, acc):
    # assert key == 'train_best_acc' or key == 'test_best_acc'
    best_acc = Config.instance().get(key)
    best_acc = max(best_acc, acc)
    Config.instance().set(key, best_acc)


def reset():
    Config.instance().set('train_class_acc', 0.0)
    Config.instance().set('test_class_acc', 0.0)
    Config.instance().set("idx", 0)
    Config.instance().set("epoch", 0)


def update(update_dict):
    Config.instance().update(update_dict)


def printall():
    print("================================")
    Config.instance().printall()
    print("================================")


class Config:
    _instance = None

    @classmethod
    def _getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls._getInstance
        return cls._instance

    def __init__(self):
        self._dict = {}

    def set(self, key, val):
        self._dict[key] = val

    def get(self, key):
        return self._dict[key]

    def update(self, update_dict):
        self._dict.update(update_dict)

    def printall(self):
        for key, val in self._dict.items():
            print('{}: {}'.format(key, val))
