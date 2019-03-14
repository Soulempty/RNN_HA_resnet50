import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import scipy.io as sio
import  torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
class RNN_HA(nn.Module):
    __factory = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    def __init__(self,depth=50,hidden_dim=1024,num_layers=1,num_model_cls=9,
                 num_vehicle_cls=776,pretrained=True):
        super(RNN_HA,self).__init__()
        if depth not in RNN_HA.__factory:
            raise KeyError("Unsupported depth:", depth)
        model=RNN_HA.__factory[depth](pretrained=pretrained)
        self.resnet=model
        self.first_gap = nn.AvgPool2d(kernel_size=7,stride =1)
        self.second_gap = nn.AvgPool2d(kernel_size=7,stride = 1)
        self.relu = nn.ReLU()
        self.attention_activation = nn.Softplus()#ln(1+exp(x))
        self.epilson = 1e-1
        self.word_embedding_dim=model.fc.in_features
        self.rnn = nn.GRU(self.word_embedding_dim, hidden_dim, num_layers, bidirectional=False)
        self.W1 = nn.Linear(hidden_dim, num_model_cls)
        init.kaiming_normal(self.W1.weight)
        self.W2 = nn.Linear(hidden_dim, num_vehicle_cls)
        init.kaiming_normal(self.W2.weight)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        init.kaiming_normal(self.W3.weight)
        self.W4 = nn.Linear(hidden_dim,self.word_embedding_dim)
        self.activation = nn.ReLU()
        init.kaiming_normal(self.W4.weight)
    def forward(self, im_feat, h0):
        for name, module in self.resnet._modules.items():
            if name == 'avgpool':
                break
            im_feat = module(im_feat)
        print('.......img_vec size:',im_feat.size())
        first_im_feat = self.first_gap(im_feat)
        first_im_feat = first_im_feat.view(-1,self.word_embedding_dim)#batchx2048
        first_im_feat = torch.unsqueeze(first_im_feat,0)##1xbatchx2048
        first_im_feat = torch.cat((first_im_feat,first_im_feat),0)#2xbatchx2048
        output, hn = self.rnn(first_im_feat, h0)
        o_model = output[0, :, :]#output:2xbatchx1024

        o_model_input = self.W3(o_model)
        o_model_input = self.relu(o_model_input)
        o_model_input = self.W4(o_model_input)
        o_model_input_expand = o_model_input.unsqueeze(2).unsqueeze(2).expand_as(im_feat)

        second_im_score = torch.mul(im_feat,o_model_input_expand)
        second_im_score = torch.sum(second_im_score,dim=1,keepdim = True)
        second_im_score = self.attention_activation(second_im_score)

        second_im_score = second_im_score + self.epilson
        second_im_score_total = torch.sum(second_im_score.view(-1,49),dim=1,keepdim=True)
        second_im_score_normalized = torch.div(second_im_score,second_im_score_total.unsqueeze(2).unsqueeze(2).expand_as(second_im_score))

        second_im_feat = torch.mul(im_feat,second_im_score_normalized.expand_as(im_feat))
        second_im_feat = self.second_gap(second_im_feat)
        second_im_feat = second_im_feat.view(-1,self.word_embedding_dim).unsqueeze(0)
        output, hn = self.rnn(second_im_feat, hn)
        o_vin = output[0, :, :]

        pred_model = self.W1(o_model)
        pred_vin = self.W2(o_vin)
        return F.log_softmax(pred_model,dim=1), F.log_softmax(pred_vin,dim=1)










