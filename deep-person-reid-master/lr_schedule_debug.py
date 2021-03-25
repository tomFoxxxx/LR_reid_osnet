from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import math
import os.path as osp
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler, RandomIdentitySampler2
from optimizers import init_optim

from IPython import embed
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=180, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=60, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")

#margin
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
#num_instance sampled per identity
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
#whether we only use hard batch triplet loss
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

model = models.init_model(name=args.arch, num_classes=751, loss={'xent', 'htri'})
optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

'''warm up与其他学习率方法的结合（基于LambdaLR）'''
config = {}
config["max_num_epochs"] = 60
warm_up_epochs = 10
lr_milestones = [20,40]

# MultiStepLR without warm up
multistep_lr = lambda epoch: 0.1**len([m for m in lr_milestones if m <= epoch])

# warm_up_with_multistep_lr
warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                            else 0.1**len([m for m in lr_milestones if m <= epoch])

# warm_up_with_step_lr
gamma = 0.1
stepsize = 20
warm_up_with_step_lr = lambda epoch: (epoch) / warm_up_epochs if epoch < warm_up_epochs \
    else gamma**((epoch-warm_up_epochs)//stepsize)

# warm_up_with_step_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
#     else gamma**( ((epoch - warm_up_epochs) /(config["max_num_epochs"] - warm_up_epochs))//stepsize*stepsize)

# warm_up_with_cosine_lr
warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
    else 0.5 * ( math.cos((epoch - warm_up_epochs) /(config["max_num_epochs"] - warm_up_epochs) * math.pi) + 1)

#scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=multistep_lr)
#scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_multistep_lr)
#scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_step_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

cur_lr_list = []

for epoch in range(60):
    #print(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()
    #if epoch == 59: print(optimizer.param_groups[-1]['lr'])
    cur_lr = optimizer.param_groups[-1]['lr']
    cur_lr_list.append(cur_lr)

x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list)
plt.show()




