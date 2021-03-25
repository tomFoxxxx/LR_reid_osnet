# -*- coding: utf-8 -*-
# @Time : 2021/2/27 19:00
# @Author : TOMFOXXXX
# @FileName: return_img_future.py
# @Software: PyCharm


from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xlwt
import pandas as pd

import torch
import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim

'''
Test
'''
#find all images in specifical folder
def find_all_img(file):
    for root, ds, fs in os.walk(file):
        for f in fs:
            if f.endswith('.jpg'):
                fullname = os.path.join(root, f)
                yield fullname

#read image
def read_image(dir):
    image = Image.open(dir).convert('RGB')
    image_root = dir
    return image, image_root

#draw image
def plt_image(img):
    # image_dir = './data/market1501/bounding_box_train/0002_c1s1_000451_03.jpg'
    # image = Image.open(image_dir).convert('RGB')
    image = np.array(img)
    print("image_shape: ", image.shape)
    print("image_dtype: ", image.dtype)
    print("image_type: ", type(image))
    plt.imshow(image)
    plt.show()

#set transform
transform_img = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load model
def load_model(root):
    model = models.init_model(name='osnet_x1_0', num_classes=751, loss={'xent'}, use_gpu=True)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Loading checkpoint from '{}'".format(root))
    checkpoint = torch.load(root)
    model.load_state_dict(checkpoint['state_dict'])
    return model

#save the output to excel
def export_excel(export):
    pf = pd.DataFrame(list(export))
    order = ['img_path','img_f']
    pf = pf[order]
    columns_map = {
    'img_path':'图片地址',
    'img_f':'图片特征'
    }
    pf.rename(columns = columns_map, inplace = True)
    #the name of excel
    file_path = pd.ExcelWriter('img_features.xlsx')
    #replace null
    pf.fillna(' ',inplace = True)
    #save
    pf.to_excel(file_path, encoding = 'utf-8', index = False)
    file_path.save()

f_list = []

model = load_model(r'D:\pycharm\LR_reid\osnet\deep-person-reid-master\log\osnet_x1_0-xent-market1501\best_model.pth.tar')
model.eval()

if __name__ == '__main__':
    start = datetime.datetime.now()
    file = './data/market1501/query'

    count_img = 0
    for pth in find_all_img(file):
        img, img_path = read_image(pth)
        '''plot when you need'''
        #plt_image(img)
        '''PIL image --> (1*C*H*W) RGB tensor'''
        img_t = transform_img(img).unsqueeze(0)
        img_t.cuda()
        img_f = model(img_t)
        img_f = img_f.data.numpy()
        f_list.append({'img_path':img_path,'img_f':img_f})
        '''print when you need'''
        # print('feature: {}  size: {}'.format(img_f,img_f.size()))
        # print(f_list)
        count_img += 1
        print('processing image nums: {}'.format(count_img))
        '''for test'''
        # if count_img >= 100:
        #     break

    '''save features'''
    export_excel(f_list)

    end = datetime.datetime.now()
    print('total_time_costs: {}seconds'.format((end - start).seconds))


