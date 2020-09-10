# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import os
import random
import re
from math import ceil

import numpy as np
import scipy.misc
from PIL import Image
import cv2

import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import generate_orientation_map_pandora as generate_orientation_map     #  Pandoradrive
# import generate_orientation_map_Livi as generate_orientation_map        #  Livi-set

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def norm_break(break_position):
    """
    归一化刹车数据
    """
    return (break_position-50.0)/50.0

def norm_steer(steer_position):
    """
    归一化方向盘数据
    """
    # tmp=int((steer_position/1000.0+3.141592454)/(2*3.141592654/181))
    tmp=steer_position/1000.0 * 180.0 / np.pi - 12.0
    # tmp = round(tmp)
    if tmp>180.0:
        tmp = 180.0
    elif tmp<-180.0:
        tmp = -180.0
    return tmp
    # if tmp<0: return 0
    # elif tmp>180: return 180
    # else: return tmp

def get_one_data(sts):
    rc={}
    rc['FrameID']=int(sts[0].split('\t')[2])
    rc['navID']=int(sts[1].split('\t')[2])
    rc['Gear_Position']=int(sts[2].split('\t')[2])
    rc['Break_Position']=int(sts[3].split('\t')[2])
    rc['Steer_Position']=int(sts[4].split('\t')[2])
    rc['Throttle_Position']=int(sts[5].split('\t')[2])
    rc['Matched_navID']=int(sts[6].split('\t')[2])
    return rc
    
# Data providers for liebao data
class Provider:
    def __init__(self, train_dir='/data/wp/code/DBnet_og/leibao_data/train/',
                 val_dir='/data/wp/code/DBnet_og/leibao_data/val/',
                 test_dir='/data/wp/code/DBnet_og/leibao_data/test',
                 camera_id=0,
                 add_lstm=True):
        self.camera_id=camera_id
        
        self.img_map = generate_orientation_map.Img_map()
        
        self.__initialize__(train_dir, val_dir, test_dir,add_lstm)
        self.read()
        self.transform()

    def __initialize__(self, train_dir, val_dir, test_dir,add_lstm):
        """
        训练用：
        X_train1：图像路径
        X_train2：点云路径，las文件，保持兼容性
        X_train3：点云路劲，pcd文件
        Y_train1：刹车位置
        Y_train2：方向盘位置
        frame_id_train：数据的frame id号
        验证用：
        X_val1：图像路径
        X_val2：点云路径，las文件，保持兼容性
        X_val3：点云路劲，pcd文件
        Y_val1：刹车位置
        Y_val2：方向盘位置
        frame_id_val：数据的frame id号
        测试用：
        X_test1：图像路径
        X_test2：点云路径，las文件，保持兼容性
        X_test3：点云路劲，pcd文件
        frame_id_test：数据的frame id号，用序号代替
        """
        self.X_train1, self.X_train2, self.X_train3 = [], [], []
        self.x_train1, self.x_train2, self.x_train3 = [], [], []
        self.Y_train1, self.Y_train2 = [], []
        self.frame_id_train = []
        
        self.X_val1, self.X_val2, self.X_val3 = [], [], []
        self.x_val1, self.x_val2, self.x_val3 = [], [], []
        self.Y_val1, self.Y_val2 = [], []
        self.frame_id_val = []
        
        self.X_test1, self.X_test2, self.X_test3 = [], [], []
        self.x_test1, self.x_test2, self.x_test3 = [], [], []
        self.frame_id_test = []
                
        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.cache_train = False
        self.cache_val = False
        self.cache_test = False
        self.add_lstm = add_lstm
    
    def read(self, filename="behavior.csv"):
        """
        Read data and labels
            :param filename: filename of labels
        """
        train_sub = glob.glob(os.path.join(self.train_dir, "*"))
        val_sub = glob.glob(os.path.join(self.val_dir, "*"))
        test_sub = glob.glob(os.path.join(self.test_dir, "*"))
        train_sub.sort(key=natural_keys)
        val_sub.sort(key=natural_keys)
        self.read_from(train_sub, filename, "train")
        self.read_from(val_sub, filename, "val")
        self.read_from(test_sub, filename, "test")
        print("magic done")
        
    def read_from(self, sub_folders, filename, description="train"):
        if description == "train":
            X_train1, X_train2, X_train3 = self.X_train1, self.X_train2, self.X_train3
            Y_train1, Y_train2 = self.Y_train1, self.Y_train2
            frame_id=self.frame_id_train
        elif description == "val":
            X_train1, X_train2, X_train3 = self.X_val1, self.X_val2, self.X_val3
            Y_train1, Y_train2 = self.Y_val1, self.Y_val2
            frame_id=self.frame_id_val
        elif description == "test":
            X_train1, X_train2, X_train3 = self.X_test1, self.X_test2, self.X_test3
            frame_id=self.frame_id_test
        else:
            raise NotImplementedError
        
        if not description == "test":
            for folder in sub_folders:
                fp=open(os.path.join(folder,'match.txt'),'r')
                data_sts=fp.readlines()
                fp.close()
                state_data=[]
                n=len(data_sts)/8
                for i in range(n):
                    state_data.append(get_one_data(data_sts[i*8+1:(i*8+8)]))
                
                for state_idx in range(n):
                    X_train1.append(os.path.join(folder,"%02d"%(self.camera_id),"%08d.jpg"%(state_data[state_idx]['Matched_navID'])))
                    X_train2.append([])
                    X_train3.append(os.path.join(folder,"pcd","%08d.pcd"%(state_data[state_idx]['Matched_navID'])))

                    Y_train1.append(norm_break(float(state_data[state_idx]['Break_Position'])))
                    Y_train2.append(norm_steer(float(state_data[state_idx]['Steer_Position'])))
                    frame_id.append(state_data[state_idx]['Matched_navID'])

        else:
            for folder in sub_folders:
                lidar_files=os.listdir(os.path.join(folder,'pcd'))
                lidar_files.sort()               
                img_files=os.listdir(os.path.join(folder,'%02d'%(self.camera_id)))
                img_files.sort()
                
                assert len(lidar_files)==len(img_files)
                for i in range(len(lidar_files)):
                    X_train1.append(os.path.join(folder, "%02d"%(self.camera_id),img_files[i]))
                    X_train2.append([])
                    X_train3.append(os.path.join(folder, "pcd",lidar_files[i]))
                    frame_id.append(i)

        if not description == "test":
            print(X_train1[:2])
            c = list(zip(X_train1, X_train2, X_train3, Y_train1, Y_train2, frame_id))
        else:
            c = list(zip(X_train1, X_train2, X_train3,frame_id))
            
        if (self.add_lstm==False and description=='train'):
            random.shuffle(c)

        if description == "train":
            self.X_train1, self.X_train2, self.X_train3, self.Y_train1, self.Y_train2, self.frame_id_train = zip(*c)
        elif description == "val":
            self.X_val1, self.X_val2, self.X_val3, self.Y_val1, self.Y_val2, self.frame_id_val = zip(*c)
        elif description == "test":
            self.X_test1, self.X_test2, self.X_test, self.frame_id_test = zip(*c)
        else:
            raise NotImplementedError

    def transform(self):
        self.X_train1, self.X_train2, self.X_train3 = np.asarray(self.X_train1), np.asarray(self.X_train2), np.asarray(self.X_train3)
        self.Y_train = np.transpose(np.asarray((self.Y_train1, self.Y_train2)))

        self.X_val1, self.X_val2, self.X_val3 = np.asarray(self.X_val1), np.asarray(self.X_val2), np.asarray(self.X_val3)
        self.Y_val = np.transpose(np.asarray((self.Y_val1, self.Y_val2)))

        self.num_test = len(self.X_test1)
        self.X_test1, self.X_test2, self.X_test3 = np.asarray(self.X_test1), np.asarray(self.X_test2), np.asarray(self.X_test3)

        self.num_train = len(self.Y_train)
        self.num_val = len(self.Y_val)
        print("train size %d"%(len(self.Y_train)))
    
    def shuffle_point(self):
        for i in range(len(self.x_train3)):
            np.random.shuffle(self.x_train3[i])
    
    def load_one_batch(self, batch_size, description='train', shape=[66, 200],
            point_num=15000, reader_type="io", add_ori=False):
        x_out1 = []
        x_out3 = []
        y_out = []
        frame_id_out=[]
        cache=False
        if description == 'train':
            if not self.cache_train:
                print ("Loading training data ...")
                for i in range(self.num_train):
                    if add_ori:
                        self.x_train1.append(self.img_map.get_dep(self.X_train1[i]))
                    else:
                        img = cv2.imread(self.X_train1[i])
                        img = img[-433:-10,:,:]
                        img = cv2.resize(img, (200,66))
                        self.x_train1.append(img/255.0)
                    self.cache_train = True
                print ("Finished loading!")

            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.X_train1)
                x_out1.append(self.x_train1[index])
                y_out.append(self.Y_train[index][1])
            self.train_pointer += batch_size
        elif description == "val":
            if not self.cache_val:
                print ("Loading validation data ...")
                for i in range(0, self.num_val):
                    if add_ori:
                        self.x_val1.append(self.img_map.get_dep(self.X_val1[i]))
                    else:
                        img = cv2.imread(self.X_val1[i])
                        img = img[-433:-10,:,:]
                        img = cv2.resize(img, (200,66))
                        self.x_val1.append(img/255.0)

                    self.cache_val = True
                print ("Finished loading!")
                
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.X_val1)
                x_out1.append(self.x_val1[index])

                y_out.append(self.Y_val[index][1])
            self.val_pointer += batch_size
        elif description == "test":
            if not self.cache_test:
                print ("Loading test data ...")
                for i in range(0, len(self.X_test1)):
                    if add_ori:
                        self.x_test1.append(self.img_map.get_dep(self.X_test1[i]))
                    else:
                        img = cv2.imread(self.X_test1[i])
                        img = img[-433:-10,:,:]
                        img = cv2.resize(img, (200,66))
                        self.x_test1.append(img/255.0)

                    self.cache_test = True
                print ("Finished loading!")
                
            for i in range(0, batch_size):
                index = (self.test_pointer + i) % len(self.X_test1)
                x_out1.append(self.x_test1[index])

            self.test_pointer += batch_size
        
        if not description == "test":
            if reader_type == "io": return np.stack(x_out1), np.stack(y_out)
            else: return np.stack(x_out1), np.stack(x_out2), np.stack(y_out)
        else:
            if reader_type == "io": return np.stack(x_out1)
            else: return np.stack(x_out1), np.stack(x_out2)

       
if __name__ == "__main__":
    data = Provider()
    print(data.X_train1[12000])
    print(len(data.X_train1))
    print(len(data.Y_train))
    print(len(data.X_val1), 'val')
    print(len(data.X_test1), 'test')
    print(data.X_train1[:5])
    print(data.Y_train2[:5])
    print(max(data.Y_train2), min(data.Y_train2), data.X_train1[data.Y_train2.index(min(data.Y_train2))])
    
    print(data.X_val1[:5])
    print(data.Y_val2[:5])
    print(max(data.Y_val2), min(data.Y_val2), data.X_val1[data.Y_val2.index(min(data.Y_val2))])
    # imgs, labels = data.load_one_batch(8, "train", reader_type="io")
    # print(imgs.shape)
    
