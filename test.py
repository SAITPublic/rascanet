"""
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
"""

from datasets.dataset import get_loader
import argparse
from models.rascanet import RaScaNet
import config
import os
import torch
from tqdm import tqdm

from utils import print_model_weight_size, print_model_ops_memory, AverageMeter

parser = argparse.ArgumentParser()
#
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--dataset', type=str, default='vww')
parser.add_argument('--dataset_path', type=str, default=os.path.join(os.path.expanduser('~'),
                                                                     'db/VWW/coco2014/'))
parser.add_argument('--rsz_w', type=int, default=120)  # 120, 240
parser.add_argument('--model_path', type=str, default='checkpoint/rascanet_105x120.pth.tar')
parser.add_argument('--split', type=int, default=5)
parser.add_argument('--early_terminate', type=int, default=0)
# attention
parser.add_argument('--skip_channel_attention', type=int, default=0)
parser.add_argument('--skip_spatial_attention', type=int, default=1)

args = parser.parse_args()
if args.early_terminate:
    args.batch_size = 1
else:
    args.batch_size = 128

# config
args = config.init(args)
for key, val in vars(args).items():
    config.set(key, val)

cur_dir = os.getcwd()
config.printall()

GPU_NUM = args.gpu_num
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

# dataset
train_loader, test_loader = get_loader(args.dataset, config.get('dataset_path'), config.get('train_path'),
                                       config.get('test_path'), args.batch_size, args.rsz_h, args.rsz_w)


def test(net, early_terminate, ops):
    net.eval()
    correct_class = 0
    total_class = 0
    et_meter = AverageMeter()
    pos_et_meter = AverageMeter()
    neg_et_meter = AverageMeter()
    et_records = []

    with torch.no_grad():
        for step, (input, cls) in enumerate(tqdm(test_loader, 0)):
            batch_size = cls.shape[0]
            input = input.cuda()
            cls = cls.cuda()

            if early_terminate:
                output, et_ratio = net.forward_early_terminate(input)
                et_meter.update(et_ratio, args.batch_size)
                et_records.append([cls.cpu().numpy()[0], et_ratio])
            else:
                _, _, output = net.forward_fast(input)

            _, class_predict = output.max(1)
            total_class += batch_size
            correct_class += class_predict.eq(cls).sum().item()

    # Accuracy
    class_acc = 100. * correct_class / total_class
    print('Test Class Acc: %.3f%% (%d/%d)' % (class_acc, correct_class, total_class))
    if early_terminate:
        if et_records:
            for item in et_records:
                if item[0] == 0:
                    neg_et_meter.update(item[1], 1)
                elif item[0] == 1:
                    pos_et_meter.update(item[1], 1)
        print('Early Termination Ratio (Avg): %.3f%% + Number of OPs: %.2fM' % (et_meter.avg, et_meter.avg * ops))
        print('Early Termination Ratio (Pos): %.3f%%' % (pos_et_meter.avg))
        print('Early Termination Ratio (Neg): %.3f%%' % (neg_et_meter.avg))


if __name__ == '__main__':
    model_path = os.path.join(cur_dir, args.model_path)
    checkpoint = torch.load(model_path)

    print('==> Building model..')
    net = RaScaNet(n_class=2)
    net.eval()
    print(net)
    print_model_weight_size(net)
    ops = print_model_ops_memory(net, iwidth=args.rsz_w, iheight=args.rsz_h)
    # run
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    test(net, args.early_terminate, ops)
