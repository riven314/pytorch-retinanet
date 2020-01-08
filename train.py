"""
Code Changed:
1. Anchor sizes and ratios
2. Image scale kept at 1 for all images (No resizing is done)
3. COCO evaluation pipeline
4. cocodataset.num_classes
5. check different backbones for their output shape
6. upload on github
"""
import os
import sys
import argparse
import collections

import numpy as np

from torchvision import transforms
import torch
import torch.optim as optim


from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer, ToTensor
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

from tensorboardX import SummaryWriter

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description = 'Simple training script for training a RetinaNet network.')
    parser.add_argument('--s', help = 'training session', type = int)
    parser.add_argument('--bs', help = 'batch size', type = int, default = 4)
    parser.add_argument('--lr', help = 'learning rate', type = float, default = 0.001)
    parser.add_argument('--save_int', help = 'interval for saving model', type = int)
    parser.add_argument('--dataset', help = 'Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help = 'Path to COCO directory')
    parser.add_argument('--csv_train', help = 'Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help = 'Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help = 'Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help = 'Resnet depth, must be one of 18, 34, 50, 101, 152', type = int, default = 50)
    parser.add_argument('--epochs', help = 'Number of epochs', type = int, default = 100)
    parser.add_argument('--use_tb', help = 'whether to use tensorboard', action = 'store_true')
    parser.add_argument('--use_aug', help = 'whether to use data augmentation', action = 'store_true')

    parser = parser.parse_args(args)
    session = parser.s
    session_dir = 'session_{:02d}'.format(session)
    assert os.path.isdir('models'), '[ERROR] models folder not exist'
    assert os.path.isdir('logs'), '[ERROR] logs folder not exist'
    model_dir = os.path.join('models', session_dir)
    logs_dir = os.path.join('logs', session_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)

    # set up tensorboard logger
    tb_writer = None
    if parser.use_tb:
        tb_writer = SummaryWriter('logs')

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        
        if parser.use_aug:
            #transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
            dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform = transforms.Compose([Normalizer(), Augmenter(), ToTensor()]))
             
        else:
            dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform = transforms.Compose([Normalizer(), ToTensor()]))

        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform = transforms.Compose([Normalizer(), ToTensor()]))
                                  #transform = transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), ToTensor()]))
                                   #transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Augmenter(), ToTensor()]))
                                     #transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size = parser.bs, drop_last = False)
    dataloader_train = DataLoader(dataset_train, num_workers = 0, collate_fn = collater, batch_sampler = sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size = parser.bs, drop_last = False)
        dataloader_val = DataLoader(dataset_val, num_workers = 0, collate_fn = collater, batch_sampler = sampler_val)

    print('# classes: {}'.format(dataset_train.num_classes))
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes = dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes = dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes = dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes = dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes = dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    # disable multi-GPU train
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr = parser.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = True)

    loss_hist = collections.deque(maxlen = 500)

    retinanet.train()
    #retinanet.module.freeze_bn() if DataParallel activated
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        # retinanet.module.freeze_bn() if DataParallel activated
        retinanet.module.freeze_bn()

        epoch_loss = []
        iter_per_epoch = len(dataloader_train)

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                assert data['img'][0].shape[0] == 3, '[ERROR] data first dim should be 3! ({})'.format(data['img'][0].shape)
                # data['img']: (B, C, H, W)
                # data['annot']: [x1, y1, x2, y2, class_id]
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # epoch starts from 0
                if (iter_num + 1) % 1 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Total loss: {:1.5f} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                        epoch_num, iter_num, float(loss), float(classification_loss), float(regression_loss), np.mean(loss_hist)
                                )
                            )
                
                # update tensorboard
                if tb_writer is not None:
                    crt_iter = (epoch_num) * iter_per_epoch + (iter_num + 1)
                    tb_dict = {
                        'total_loss': float(loss),
                        'classification_loss': float(classification_loss),
                        'regression_loss': float(regression_loss)
                    }
                    tb_writer.add_scalars('session_{:02d}/loss'.format(session), tb_dict, crt_iter)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))
        if (epoch_num + 1) % parser.save_int == 0:
            # retinanet (before DataParallel): <class 'retinanet.model.ResNet'>, no self.module
            # retinanet (after DataParallel): <class 'torch.nn.parallel.data_parallel.DataParallel>, self.module available
            # retinanet.module (after DataParallel): <class 'retinanet.model.ResNet'>
            torch.save(retinanet.module.state_dict(), os.path.join(model_dir, 'retinanet_s{:02d}_e{:03d}.pth'.format(session, epoch_num)))

    if parser.use_tb:
        tb_writer.close()

    retinanet.eval()
    torch.save(retinanet.module.state_dict(), os.path.join(model_dir, 'retinanet_s{:02d}_e{:03d}.pth'.format(session, epoch_num)))


if __name__ == '__main__':
    main()
