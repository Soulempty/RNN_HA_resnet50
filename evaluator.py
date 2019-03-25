from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import os
import torch
from metrics import cmc, mean_ap
from torch.autograd import Variable
from utils import AverageMeter
from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def extract_feat(model, inputs):
    use_cuda=True
    hidden_dim=1024
    num_layers=1
    h0 = 0.0 * torch.randn(num_layers,len(inputs), hidden_dim)
    if use_cuda:
        inputs = Variable(inputs.cuda())
        h0=Variable(h0.cuda())
    else:
        inputs = Variable(inputs)
        h0=Variable(h0)
    model.eval()
    tmp = model(inputs,h0)
    outputs = tmp[2]
    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, print_freq=20):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    
    labels = OrderedDict()
    features = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, vehids,cams) in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs = extract_feat(model, imgs)
        for fname, output, vehid in zip(fnames, outputs, vehids):
            labels[fname]= vehid.item()

            features[fname] =output
            

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(query_features, gallery_features, query=None, gallery=None,dist='eu'):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(1) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist
    
    x = torch.cat([query_features[os.path.basename(f)].unsqueeze(0) for f, _,_ in query], 0)
    y = torch.cat([gallery_features[os.path.basename(f)].unsqueeze(0) for f, _,_ in gallery], 0)
    m, n = x.size(0), y.size(0)
    #欧氏距离计算步骤
    if dist!='cosin':
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())#(x1-y1)**2+(x2-y2)**2==x1**2+x2**2+y1**2+y2**2-2*(x1*y1+x2*y2)
    else:
        x = x.view(m, -1)
        y = y.view(n, -1)
        x_norm=torch.norm(x,2,1,True).expand_as(x)
        y_norm=torch.norm(y,2,1,True).expand_as(y)
        x_n=x.div(x_norm)
        y_n=y.div(y_norm)
        dist=x_n.mm(y_n.t())#m*n (-1,1)
        dist=0.5+dist*0.5
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10),
                 dataset="veri"):
    name={"veri":"VeRi","vihicleid":"VehicleID"}
    if query is not None and gallery is not None:
        query_ids = [label for _, label,_ in query]
        gallery_ids = [label for _, label,_ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    dataName=name[dataset]
    print("------------------------------")
    print("The result of the dataset({})".format(dataName))
    print('Mean AP: {:4.1%}'.format(mAP))
    
    cmc_scores=cmc(distmat, query_ids, gallery_ids,query_cams, gallery_cams)

    print('CMC Scores:{:4.1%}'
          .format(cmc_scores[0]))
    for k in cmc_topk:
        print('  top-{0:}: {1:4.1%}'
              .format(k, cmc_scores[k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores[0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery):

        print('extracting query features\n')
        query_features, _ = extract_features(self.model, query_loader)
        print('extracting gallery features\n')
        gallery_features, _ = extract_features(self.model, gallery_loader)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery,)
