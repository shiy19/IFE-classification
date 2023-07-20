# -*- coding:utf-8 -*-
import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import pandas as pd

class Spectral_MDC(Dataset):
    def __init__(self, setname, aug = True, large_img = True):
        # locate the dir
        # label1 for None, G, A, M, None, label2 for None, K, L
        self.large_img = large_img
        self.data, self.label1, self.label2, self.label = [], [], [], []
        # self.attr = attr
        rootpath = osp.join(os.getcwd(),'..')
        if setname == 'train': 
            df = pd.read_csv(osp.join(rootpath,'label','train.csv'))
        elif setname == 'test':            
            df = pd.read_csv(osp.join(rootpath,'label','test.csv')) 
        else:
            raise ValueError("No such split!")
        for i in range(df.shape[0]):
            id, label1, label2, label = df.loc[i,'id'],df.loc[i,'label1'], df.loc[i,'label2'], df.loc[i,'label']
            self.label1.append(label1)
            self.label2.append(label2)
            self.data.append(id)            
            self.label.append(label)
           
        # Transformation
        image_size = 224 if self.large_img else 32
        self.image_size = image_size
        transforms_list = [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            ] 

        if setname == 'train' and aug == True:
            # Transformation
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.5, 0.5, 0.5]),
                                     np.array([0.5, 0.5, 0.5]))]) 
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.5, 0.5, 0.5]),
                                     np.array([0.5, 0.5, 0.5]))])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        im, label1, label2, label = self.data[i], self.label1[i], self.label2[i], self.label[i]
        image = self.transform(Image.open(im).convert('RGB'))
        return image, label1, label2, label

    def getimg(self, i):
        im, label = self.data[i], self.label[i]
        return im, label

    def getimage(self,i):
        im_path = self.data[i]
        pil_image = Image.open(im_path)
        torch_img = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor()
        ])(pil_image)
        return torch_img

def test():
    trainset = Spectral_MDC('train')
    valset = Spectral_MDC('test')
    trainset[0]
if __name__ == "__main__":
    test()