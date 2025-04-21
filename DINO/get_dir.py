#import cv2
import os
import numpy as np
import torch
from shutil import move,copy,rmtree

org_path='/home/16t1/granuloma/fei-ryz_svs/data_mycobacteria/val/neg/'
save_path='/home/16t1/granuloma/fei-ryz_svs/data_mycobacteria/val_dir/neg/'
data_list=os.listdir(org_path)
name_list=[]
for i in range(len(data_list)):
    pth=data_list[i].split('.')[0].split('_')[0]

    if data_list[i].split('.')[-1]=='png':
        if not os.path.exists(save_path+pth):
            os.makedirs(save_path+pth)

        copy(org_path+data_list[i],save_path+pth)
    else:
        rmtree(org_path+data_list[i])
        print(pth)

