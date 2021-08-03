"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

from datasets.vww_dataset import VWWDataset
from preprocess import data_transforms_vww, cls_bbox_transform
from torch.utils.data import DataLoader


def get_loader(dataset, dataset_path, train_path, test_path, batch_size, rsz_h, rsz_w, randcrop=None):
    print('==> Preparing data..')
    train_loader = None
    test_loader = None
    if dataset == 'vww':
        train_transform, test_transform = data_transforms_vww(rsz_h, rsz_w, randcrop)
        train_dataset = VWWDataset(root=dataset_path + 'all', annotation_file=train_path, transform=train_transform,
                                   target_transform=cls_bbox_transform)
        test_dataset = VWWDataset(root=dataset_path + 'all', annotation_file=test_path, transform=test_transform)
        #
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    elif dataset == 'voc':
        print('currently this version does not support pascal voc dataset')
        exit()

    return train_loader, test_loader
