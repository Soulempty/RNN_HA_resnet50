import numpy as np
import torch
from PIL import Image
import random
import os
import glob
from img_proc import *
import scipy.misc as m
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Normalize,Resize
from torch.utils.data import Dataset



class MyTransform(Dataset):
    def __init__(self,data_path,augmentations=None,rgb_mean=(128,128,128)): 
        
        self.mean = np.array(rgb_mean)
        self.augmentations=augmentations
        self.images = []
        self.data_path=data_path+'/*.jpg'
        files=sorted(glob.glob(self.data_path))
        label=sorted(set([int(os.path.basename(f).split("_")[0]) for f in files]))      
        l_len=len(label)
        
        self.vclass_to_id={label[i]: i for i in range(l_len)}
        for f in files:
            img_name=os.path.basename(f)
            label=int(img_name.split("_")[0])
            self.images.append((f,self.vclass_to_id[label]))

    def __len__(self):
        return len(self.images)  
    def __getitem__(self, i):
        # do something to both images and labels
        img, label = self.images[i]
        img=Image.open(img)
        if self.augmentations is not None:
            img = self.augmentations(img)
            
        return img, label
class MydataLoader(Dataset):
    def __init__(self,data_path,txt_path,augmentations=None,rgb_mean=(128,128,128)): 
        
        self.mean = np.array(rgb_mean)
        self.augmentations=augmentations
        self.images = []
        self.data_path=data_path
        self.txt_path=txt_path
        imgs=open(self.txt_path).readlines()
        model_l=sorted(set([int(img.split()[2]) for img in imgs]))
        m_len=len(model_l)
        print("model cls:",m_len)
        self.mclass_to_id={model_l[i]: i for i in range(m_len)}
        veh_l=sorted(set([int(img.split()[1]) for img in imgs]))
        v_len=len(veh_l)
        print("veh cls:",v_len)
        self.vclass_to_id={veh_l[i]: i for i in range(v_len)}
        for img in imgs:
            img_path=os.path.join(self.data_path,img.split()[0])
            veh_label=int(img.split()[1])
            model_label=int(img.split()[2])
            self.images.append((img_path,self.mclass_to_id[model_label],self.vclass_to_id[veh_label]))
    def __len__(self):
        return len(self.images)  
    def __getitem__(self, i):
        # do something to both images and labels
        img, model_label,veh_label = self.images[i] 
        img=Image.open(img).convert('RGB')
        if self.augmentations is not None:
            img = self.augmentations(img)
        veh_label = veh_label
        model_label = model_label

        return img, model_label,veh_label    



