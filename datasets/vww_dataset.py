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
from PIL import Image

from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO


class VWWDataset(VisionDataset):

    def __init__(self, root, annotation_file, transform, target_transform=None):
        super(VWWDataset, self).__init__(root, None, transform, target_transform)

        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        annotations = self.coco.loadAnns(ann_ids)
        cls = annotations[0]['category_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        bboxes = []
        for annotation in annotations:
            if annotation.get('bbox'):
                bbox = annotation.get('bbox')
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1,y1,x2,y2
                bboxes.append(bbox)

        item = {'img': img, 'bboxes': bboxes, 'cls': cls}
        if self.transform is not None:
            item = self.transform(item)
            img = item.get('img')
            bboxes = item.get('bboxes')

        if self.target_transform is not None:
            cls, _ = self.target_transform(cls, bboxes, img.shape[1] * img.shape[2], th=0.005)
        return img, cls

    def __len__(self):
        return len(self.ids)
