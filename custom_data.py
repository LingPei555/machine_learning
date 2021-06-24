from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import random

class custom_dset_I():
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None,
                 loader=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

            #************  ANN I **************#              
            self.target_list = [
                np.loadtxt(os.path.join(img_path, i.split()[0]))*np.array([1, 1, 1, 0.01]) for i in lines
            ]
            #self.img_list = [ list(map(float, i.split()[5:]))*np.array([0.001, 1, 1, 1, 1])  for i in lines] #L Q  SRF Area Freq 
            self.img_list = [ (list(map(float, i.split()[5:8]))+list(map(float, i.split()[9:])))*np.array([0.001, 1, 1, 1])  for i in lines] #L Q Freq Area 
            
            self.s11_list = [float(i.split()[1]) for i in lines]

        self.img_transform = img_transform

    def __getitem__(self, index):
        target = self.target_list[index]
        img = self.img_list[index]

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, target  

    def __len__(self):
        return len(self.s11_list)
    

class custom_dset_II():
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None,
                 loader=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()           
            #************ ANN II **************#
            self.img_list = [
                list(np.loadtxt(os.path.join(img_path, i.split()[0]))*np.array([1, 1, 1, 0.01]))+[float(i.split()[7])] for i in lines 
            ]# W S T D F
            #self.target_list = [ list(list(map(float, i.split()[1:-3]))*np.array([1, 1, 1, 1, 0.001, 1]))  for i in lines] # S L Q
            self.target_list = [ list(list(map(float, i.split()[5:-3]))*np.array([0.001, 1]))  for i in lines] # L Q
            #self.target_list = [ list(list(map(float, i.split()[1:5]))*np.array([1, 1, 1, 1]))  for i in lines] # S
            #self.target_list = [ list(list(map(float, i.split()[1:9]))*np.array([1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]))  for i in lines] # S sign
            #self.target_list = [ list(list(map(float, i.split()[5:9]))*np.array([10, 10, 10, 10]))  for i in lines] # sign
            #self.target_list = [ list(list(map(float, i.split()[7:9]))*np.array([1, 1]))  for i in lines] # S21 sign
            
            
            self.s11_list = [float(i.split()[1]) for i in lines]

        self.img_transform = img_transform

    def __getitem__(self, index):
        target = self.target_list[index]
        img = self.img_list[index]

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, target  

    def __len__(self):
        return len(self.s11_list)


    
def collate_fn(batch):
    img,target = zip(*batch)
    return img, target