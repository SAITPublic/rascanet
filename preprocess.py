"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import torch
import config


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def cls_bbox_transform(cls, bboxes, area, th=0.005):
    # check target after transform
    new_bboxes = []
    for bbox in bboxes:
        [bx1, by1, bx2, by2] = bbox
        bbox_area = (bx2 - bx1) * (by2 - by1)
        if bbox_area >= th * area:
            new_bboxes.append(bbox)

    cls = 1 if len(new_bboxes) else 0

    return cls, new_bboxes


def data_transforms_vww(rsz_h, rsz_w, randcrop=None):
    min_bright = 1. - (32. / 255.)
    max_bright = 1. + (32. / 255.)

    train_transform = transforms.Compose([
        RandomCrop((rsz_h, rsz_w), randcrop),
        Resize((rsz_h, rsz_w)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=(min_bright, max_bright), saturation=(0.5, 1.5)),
        ToTensor(),
        transforms.Lambda(Normalize),
    ])

    test_transform = transforms.Compose([
        Resize((rsz_h, rsz_w)),
        ToTensor(),
        transforms.Lambda(Normalize),
    ])

    return train_transform, test_transform


def data_transforms_voc(rsz_h, rsz_w):
    min_bright = 1. - (32. / 255.)
    max_bright = 1. + (32. / 255.)
    randcrop = config.get('randcrop')

    train_transform = transforms.Compose([
        RandomCrop((rsz_h, rsz_w), randcrop),
        Resize((rsz_h, rsz_w)),
        RandomHorizontalFlip(),
        ColorJitter(brightness=(min_bright, max_bright), saturation=(0.5, 1.5)),
        ToTensor(),
        transforms.Lambda(Normalize),
    ])

    test_transform = transforms.Compose([
        Resize((rsz_h, rsz_w)),
        ToTensor(),
        transforms.Lambda(Normalize),
    ])

    return train_transform, test_transform


class RandomCrop(object):

    def __init__(self, target_size, crop_ratios=[1.0]):
        self.target_size = target_size
        self.crop_ratios = crop_ratios

    @staticmethod
    def get_params(img, crop_ratio, target_size):

        rsz_h, rsz_w = target_size
        w, h = _get_image_size(img)
        tw, th = int(w * crop_ratio), int(h * crop_ratio)
        if tw <= rsz_w:
            tw = w
        if th <= rsz_h:
            th = h

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, item):
        crop_ratio = random.choice(self.crop_ratios)
        # img
        img = item['img']
        h_s, w_s, h, w = self.get_params(img, crop_ratio, self.target_size)
        img = F.crop(img, h_s, w_s, h, w)
        item['img'] = img

        # bbox
        if item['bboxes']:
            for idx, bbox in enumerate(item['bboxes']):
                [bx1, by1, bx2, by2] = bbox
                ow = bx2 - bx1
                oh = by2 - by1
                dw = bx1 - w_s
                dh = by1 - h_s
                bx1 = min(max(dw, 0), w)
                by1 = min(max(dh, 0), h)
                bx2 = max(0, min(bx1 + ow - max(-dw, 0), w))
                by2 = max(0, min(by1 + oh - max(-dh, 0), h))

                item['bboxes'][idx] = [bx1, by1, bx2, by2]

        return item


class Resize(transforms.Resize):

    def __call__(self, item):
        # img
        img = item['img']
        ow, oh = img.size  # width, height
        img = F.resize(img, self.size, self.interpolation)
        item['img'] = img

        # bbox
        if item['bboxes']:
            for i, bbox in enumerate(item['bboxes']):
                nh, nw = self.size  # height, width
                [bx1, by1, bx2, by2] = bbox
                h_ratio = nh / oh
                w_ratio = nw / ow

                bx1 = max(bx1 * w_ratio, 0)
                by1 = max(by1 * h_ratio, 0)
                bx2 = min(bx2 * w_ratio, nw)
                by2 = min(by2 * h_ratio, nh)

                item['bboxes'][i] = [bx1, by1, bx2, by2]

        return item


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, item):

        img = item['img']
        w, h = img.size  # width, height

        if random.random() < self.p:
            img = F.hflip(img)
            item['img'] = img
            # bbox
            if item['bboxes']:
                for i, bbox in enumerate(item['bboxes']):
                    [bx1, by1, bx2, by2] = bbox
                    center = w / 2
                    min_x = min(2 * center - bx1, 2 * center - bx2)
                    max_x = max(2 * center - bx1, 2 * center - bx2)
                    bx1 = min_x
                    bx2 = max_x
                    item['bboxes'][i] = [bx1, by1, bx2, by2]

        return item


class ColorJitter(transforms.ColorJitter):
    # stems from transforms.ColorJitter

    def __call__(self, item):
        img = item['img']
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        item['img'] = transform(img)

        return item


class ToTensor(transforms.ToTensor):

    def __call__(self, item):
        img = item['img']
        img = F.to_tensor(img)
        item['img'] = torch.clamp(img, 0.0, 1.0)

        return item


def Normalize(item):
    img = item['img']
    item['img'] = torch.round(((img - 0.5) * 2.0) * 127.0)

    return item
