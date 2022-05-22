import torch 
import torch.nn as nn 
import torch.nn.functional as F
from utils.resize import *

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y * x

class RCAB(nn.Module):
    def __init__(self, args):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv3d(args.N_feats, args.N_feats, kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1], bias=True))
            if args.rcab_bn: modules_body.append(nn.BatchNorm3d(args.N_feats))
            if i == 0: modules_body.append(nn.LeakyReLU())
        self.ca = CALayer(args.N_feats, args.reduction)
        self.body = nn.Sequential(*modules_body)
        self.T = args.N_lrs

    def forward(self, x):
        res = self.body(x)
        res = spatial_squeeze(res)
        res = self.ca(res)
        res = spatial_expand(res, self.T)
        res = res + x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, args):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [RCAB(args) for _ in range(args.N_resblocks)]
        modules_body.append(nn.Conv3d(args.N_feats, args.N_feats, kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1], bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


class LEM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(args.N_feats, args.N_heads, dropout=0.0, bias=True)
        self.tnorm1 = nn.BatchNorm3d(args.N_lrs)
        self.mha2 = nn.MultiheadAttention(args.N_feats, args.N_heads, dropout=0.0, bias=True)
        self.tnorm2 = nn.BatchNorm3d(args.N_lrs)
        self.rg1 = ResidualGroup(args)
        self.cnorm1 = nn.BatchNorm3d(args.N_feats)
        self.T = args.N_lrs
    
    def forward(self, x, q):
        S = x.size(3)

        x = spatial_expand(x, self.T)
        res = self.rg1(x)
        x = x + res 
        x = self.cnorm1(x)
        x = spatial_squeeze(x)

        q = spatial_to_mha(q, self.T)
        res = spatial_to_mha(x, self.T)
        res = self.mha1(res, res, res)[0]
        res = mha_to_spatial(res, S)
        x = x + res 
        x = from_spatial(x, self.T)
        x = self.tnorm1(x)
        x = to_spatial(x, dim=5)

        res = spatial_to_mha(x, self.T)
        res = self.mha2(q, res, res)[0]
        res = mha_to_spatial(res, S)
        x = x + res 
        x = from_spatial(x, self.T)
        x = self.tnorm2(x)
        x = to_spatial(x, dim=5)
    
        return x


class QEM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv3d(args.N_feats, args.N_feats, kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv2 = nn.Conv3d(args.N_feats, args.N_feats, kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1])
        self.cnorm = nn.BatchNorm3d(args.N_feats)
        self.T = args.N_lrs
    
    def forward(self, x):
        S = x.size(3)

        x = spatial_expand(x, self.T)
        res = self.conv1(x)
        res = self.conv2(res)
        x = x + res 
        x = self.cnorm(x)
        x = spatial_squeeze(x)
    
        x = F.sigmoid(x)

        return x


class QANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.init = nn.Conv3d(1, args.N_feats, kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1])
        lems = [LEM(args) for _ in range(args.N_modules)]
        qems = [QEM(args) for _ in range(args.N_modules)]
        self.lems = nn.Sequential(*lems)
        self.qems = nn.Sequential(*qems)
        self.last = nn.Conv2d(args.N_feats, 9, kernel_size=3, stride=1, padding=1)
        self.T = args.N_lrs
    
    def forward(self, x, q):

        x = x.unsqueeze(1)
        q = q.unsqueeze(1)

        x = self.init(x)
        q = self.init(q)

        x = spatial_squeeze(x)
        q = spatial_squeeze(q)
        
        x_o = x + 0.0
        for lemt, qemt in zip(self.lems._modules.items(), self.qems._modules.items()):
            _, lem = lemt
            _, qem = qemt
            x = lem(x, q)
            q = qem(q)
        x = x + x_o

        x = from_spatial(x, self.T)
        q = from_spatial(q, self.T)

        x = x * q
        x = torch.sum(x, dim=1)
        q = torch.sum(q, dim=1)
        x = torch.div(x, q)

        x = self.last(x)
        x = F.pixel_shuffle(x, 3)
        return x