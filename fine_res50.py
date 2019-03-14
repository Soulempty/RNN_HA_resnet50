import torch
import time
from torch.autograd import Variable
from torch import nn
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models 
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image
from resnet import *
import os
import shutil
from dataloader import *
from img_proc import Compose,Self_Scale,RandomHorizontallyFlip

import argparse
parser=argparse.ArgumentParser(description='fintune Training')
parser.add_argument('--training_data',default='./VeRi', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', default=False, type=bool, help='evaluate or not')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--depth', default='resnet50', help='which depth of resnet')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--epoches', default=10, type=int, help='max epoch for retraining')
parser.add_argument('--save_folder', default='./weight/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
55.02
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
rgb_means=[0.428,0.427,0.429]
rgb_std=[0.216,0.214,0.216]
normalizer = transforms.Normalize(mean=rgb_means,
                             std=rgb_std)
num_classes=576
depth=args.depth
__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101}


model=__factory[depth](classes=num_classes)
print("Printing net...")
print(model)
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.ngpu>1:
    model=nn.DataParallel(model,device_ids=list(range(args.ngpu)))
if args.cuda:
    model.cuda()
    cudnn.benchmark = True
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)
criterion = nn.CrossEntropyLoss().cuda()


def train(args,model,optimizer,criterion): 
    batch_size = args.batch_size
    data_path=args.training_data
    save_folder=args.save_folder
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    best_predict=0
    epoch = 0 + args.resume_epoch 
    print("Loading dataset...")
    train_data=MyTransform(data_path,Compose([Self_Scale(224),RandomHorizontallyFlip(),
            transforms.ToTensor(),
            normalizer,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    val_data=MyTransform(data_path,Compose([Self_Scale(224),
            transforms.ToTensor(),
            normalizer,
        ]))
    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size//2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    model.train()
    epoch_size = math.ceil(len(train_data) / batch_size)
    max_iter = args.epoches * epoch_size

    stepvalues = (4 * epoch_size, 8 * epoch_size)
    step_index = 0

    if args.start_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    end = time.time()
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            
            batch_iterator = iter(train_loader)
            if (epoch % 2 == 0 and epoch <5 and epoch>0):
                prec1 = validate(val_loader, model, criterion)
                is_best=prec1>best_predict
                best_predict = max(prec1, best_predict)
                save_checkpoint({
                       'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'best_prec1': best_predict,
                       'optimizer' : optimizer.state_dict(),
                        },is_best,save_folder)
            epoch += 1

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer,epoch,step_index,iteration,epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = Variable(targets.cuda(async=True))
        else:
            images = Variable(images)
            targets = Variable(targets)

        # forward
        out = model(images)
        
        # backprop
        optimizer.zero_grad()
        loss = criterion(out, targets)
        prec1, prec5 = accuracy(out.data, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        loss.backward()
        optimizer.step()
        print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, iteration % epoch_size, len(train_loader),loss=losses, top1=top1, top5=top5))
    prec1 = validate(val_loader, model, criterion)
    is_best=prec1>best_predict
    best_predict = max(prec1, best_predict)
    save_checkpoint({
                       'epoch': epoch + 1,
                       'state_dict': model.state_dict(),
                       'best_prec1': best_predict,
                       'optimizer' : optimizer.state_dict(),
                        },is_best,save_folder)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, targets) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                images = Variable(images.cuda())
                targets = Variable(targets.cuda(async=True))
            else:
                images = Variable(images)
                targets = Variable(targets)
        # compute output
        output = model(images)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, folder,filename='checkpoint.pth.tar'):
    save_path=folder+filename
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path,folder+'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer,epoch,step_index,iteration,epoch_size):
    """Sets the learning rate"""
    if epoch < 0:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (0.1 ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    train(args,model,optimizer,criterion)
    
