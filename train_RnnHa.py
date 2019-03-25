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
from PIL import Image
from RNN_HA import RNN_HA  
from dataloader import MydataLoader
from img_proc import Compose,Self_Scale,RandomHorizontallyFlip

import argparse
parser=argparse.ArgumentParser(description='fintune Training')
parser.add_argument('--training_data',default='./VeRi', help='Training dataset directory')
parser.add_argument('--txt_path',default='./train_label.txt', help='the list of imgs and its label')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',default=False,type=bool,help='eva or not')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--depth', default=50, type=int,help='which depth of resnet')
parser.add_argument('--res_resume', default='', type=str, metavar='PATH',help='resume pretrain resnet for retraining')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--epoches', default=100, type=int, help='max epoch for retraining')
parser.add_argument('--save_folder', default='./models/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

if args.res_resume:
    if os.path.isfile(args.res_resume):
        checkpoint=torch.load(args.res_resume)
        model_dict=model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
if args.resume:
    if os.path.isfile(args.resume):
        checkpoint=torch.load(args.resume)
        model.load_state_dict(checkpoint)
if args.ngpu>1:
    model=nn.DataParallel(model,device_ids=list(range(args.ngpu)))
if args.cuda:
    model.cuda()
    cudnn.benchmark = True

optimizer = optim.RMSprop(filter(lambda  p: p.requires_grad, model.parameters()), lr=args.lr)

def train(args,model,optimizer):
    
    batch_size = args.batch_size
    data_path=args.training_data
    txt_path=args.txt_path
    epoches=args.epoches
    start_epoch=args.start_epoch
    train_loss = 0
    correct_model = 0
    correct_veh = 0
    print("Loading dataset...")
    train_data=MydataLoader(data_path,txt_path,Compose([Self_Scale(224),
                           transforms.ToTensor(),
                           normalizer]))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    model.train()
    batch_sum=1
    lr_dis=0
    for epoch in range(start_epoch,epoches+1):
        if epoch<10:
            lr_dis=args.lr*(epoch/10)
            optimizer = optim.RMSprop(filter(lambda  p: p.requires_grad, model.parameters()), lr=lr_dis)
        if epoch==20 or epoch %40==0:
            lr_dis=args.lr/10**(epoch/20)
            optimizer = optim.RMSprop(filter(lambda  p: p.requires_grad, model.parameters()), lr=lr_dis)
        train_batch=1
        for sample in train_loader:
            h0 = 0.0 * torch.randn(num_layers,len(sample[0]), hidden_dim)
            img,model_label,veh_label=sample
            if args.cuda:
                img = Variable(img.cuda())
                h0=Variable(h0.cuda())
                model_label = Variable(model_label.cuda())
                veh_label = Variable(veh_label.cuda())
            else:
                img = Variable(img)
                h0=Variable(h0)
                model_label = Variable(model_label)
                veh_label = Variable(veh_label)
            output_model, output_veh ,_ = model(img, h0)
            optimizer.zero_grad()
            loss_model = F.nll_loss(output_model, model_label)
            loss_veh = F.nll_loss(output_veh, veh_label)
            pred_model = output_model.data.max(1)[1]  # get the index of the max log-probability
            correct_model += pred_model.eq(model_label.data).cpu().sum().float()/batch_size
            pred_veh = output_veh.data.max(1)[1]  # get the index of the max log-probability
            correct_veh += pred_veh.eq(veh_label.data).cpu().sum().float() / batch_size
            
            loss = loss_model + loss_veh
            loss.backward()
            optimizer.step()
            train_loss += loss.item()/batch_size
            print('batch: [{0}/{1}]\t'
                  'loss: {2:.4f} '
                  'model:{3:.3f} '
                  'vechicle:{4:.3f}'
                  .format(train_batch,epoch,train_loss/batch_sum,correct_model/batch_sum,correct_veh/batch_sum))
            train_batch+=1
            batch_sum+=1
        if not os.path.exists('./log'):
            os.mkdir('./log')
        log = open('./log/train.txt', 'a+')
        log.write("The " + str(epoch) + "-th epoch: model accuracy is " + str(100. * correct_model/batch_sum) + ", veh accuracy is " + str(100. * correct_veh/batch_sum) +"\n")
        if epoch % 40 == 0 or epoch==100:
            saveName = args.save_folder+'model_epoch_' + str(epoch)+'.pth'
            torch.save(model.state_dict(), saveName)


if __name__=='__main__':
    train(args,model,optimizer)
