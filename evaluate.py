import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
from evaluator import Evaluator
from PIL import Image
from RNN_HA import RNN_HA  
from dataloader import VeRiTransform,VeRi
from img_proc import Compose,Self_Scale,RandomHorizontallyFlip


import argparse
parser=argparse.ArgumentParser(description='Data testing')
parser.add_argument('--data_path',default='./VeRi', help='The directory of query files path')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')

parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--depth', default=50, type=int,help='which depth of resnet')

parser.add_argument('--resume', default='', type=str, metavar='PATH',help='resume net for retraining')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rgb_means=[0.428,0.427,0.429]
rgb_std=[0.216,0.214,0.216]
normalizer = transforms.Normalize(mean=rgb_means,
                             std=rgb_std)
hidden_dim=1024
num_classes=575
num_models=9
num_layers=1
depth=args.depth
model = RNN_HA(depth=depth,hidden_dim=hidden_dim,num_model_cls=num_models,
                 num_vehicle_cls=num_classes,num_layers=num_layers)
print("Printing net...")
print(model)

if args.resume:
    if os.path.isfile(args.resume):
        print('load model...')
        checkpoint=torch.load(args.resume)
        model.load_state_dict(checkpoint)

if args.cuda:
    model.cuda()
    cudnn.benchmark = True  
    
def evaluate(args,model):
    
    batch_size = args.batch_size
    data_path=args.data_path
    eva=Evaluator(model)
    
    print("Loading dataset...")
    dataset=VeRi(data_path)
    transformer=Compose([Self_Scale(224),transforms.ToTensor(),normalizer,])
    
    query_loader=torch.utils.data.DataLoader(VeRiTransform(dataset.query,transformer),
                             batch_size=batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True)        
    gallery_loader=torch.utils.data.DataLoader(VeRiTransform(dataset.gallery,
                    transformer),
                    batch_size=batch_size, num_workers=args.num_workers,
                    shuffle=False, pin_memory=True)
    eva.evaluate(query_loader,gallery_loader,dataset.query,dataset.gallery)

if __name__=='__main__':
    evaluate(args,model)

