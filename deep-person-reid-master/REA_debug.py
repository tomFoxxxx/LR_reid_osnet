# -*- coding: utf-8 -*-
# @Time : 2021/3/8 16:27
# @Author : TOMFOXXXX
# @FileName: REA_debug.py.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time : 2021/2/27 19:00
# @Author : TOMFOXXXX
# @FileName: return_img_future.py
# @Software: PyCharm


from __future__ import print_function, absolute_import
import numpy as np
import torch
import torchvision
import transforms as T
#import torchvision.transforms as T
import PIL
import PIL.Image as Image
import matplotlib.pyplot as plt


def read_image(dir):
    image = Image.open(dir).convert('RGB')
    image_root = dir
    return image, image_root

def plt_image(img):
    # image_dir = './data/market1501/bounding_box_train/0002_c1s1_000451_03.jpg'
    # image = Image.open(image_dir).convert('RGB')
    image = np.array(img)
    print("image_shape: ", image.shape)
    print("image_dtype: ", image.dtype)
    print("image_type: ", type(image))
    plt.imshow(image)
    plt.show()

'''
torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

value – erasing value. Default is 0. If a single int, it is used to erase all pixels. 
If a tuple of length 3, it is used to erase R, G, B channels respectively. 
If a str of ‘random’, erasing each pixel with random values.
'''
transform_img =  T.Compose([
        T.Resize((256,128)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256,128]),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.4), ratio=(0.3, 3.33))
        #torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.4), ratio=(0.3, 3.33), value=(0.4914, 0.4822, 0.4465))
        #torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.4), ratio=(0.3, 3.33), value=1)
        #torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.4), ratio=(0.3, 3.33), value=2)
        #torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.4), ratio=(0.3, 3.33), value=12)
        #torchvision.transforms.RandomErasing(p=1, scale=(0.02, 0.4), ratio=(0.3, 3.33), value='random')
        T.RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465)),
    ])

if __name__ == '__main__':
    pth = r'D:\pycharm\LR_reid\osnet\deep-person-reid-master\data\market1501\bounding_box_train\0002_c1s1_000451_03.jpg'
    img, img_path = read_image(pth)
    img_trans = transform_img(img)
    print(type(img_trans),img_trans.dtype,img_trans.shape,sep='\t')
    #plt_image(img_trans)
    image = PIL.Image.fromarray(torch.clamp(img_trans * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
    #image = torchvision.transforms.functional.to_pil_image(img_trans)  # Equivalently way
    plt_image(image)





