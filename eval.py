"""
print out predicted images
"""
import os
import sys
import argparse
import collections

import cv2
import numpy as np

from torchvision import transforms
import torch
import torch.optim as optim


from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer, ToTensor
from torch.utils.data import DataLoader

model_path = os.path.join('models', 'session_01', 'retinanet_s01_e099.pth')
coco_path = os.path.join('..', 'simulated_data', 'coco_format', 'mini_easy', 'coco')
write_dir = os.path.join('logs', 'session_01', 'train_results')
assert os.path.isfile(model_path), '[ERROR] model weights not exist!'
assert os.path.isdir(coco_path), '[ERROR] COCO dataset not exist!'

dataset = CocoDataset(coco_path, set_name='train2017', transform = transforms.Compose([Normalizer(), ToTensor()]))
sampler = AspectRatioBasedSampler(dataset, batch_size = 1, drop_last = False)
dataloader = DataLoader(dataset, num_workers = 0, collate_fn = collater, batch_sampler = sampler)

retinanet = model.resnet50(num_classes = dataset.num_classes(), pretrained=True)
retinanet = retinanet.cuda()
retinanet.load_state_dict(torch.load(model_path))
print('loaded model weights: {}'.format(model_path))
retinanet.eval()

data = next(iter(dataloader))
img = data['img'].cuda().float()
annot = data['annot'].cuda().float()
# preds: list of 3 items
#  - list of scores (0.05 as threshold)
#  - list of classes (start from index 0, index 0 is NOT background index)
#  - list of bbox location ([xmin, ymin, xmax, ymax])
preds = retinanet(img)

