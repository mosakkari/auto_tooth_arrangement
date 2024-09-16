import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from chamfer_dist import ChamferDistanceL1
import copy
import numpy as np
import math
from functools import partial
import torch
import timm.models.vision_transformer
from models.pointnet import PointNet
from timm.models.vision_transformer import PatchEmbed, Block

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = PointNet(first_dim=100,second_dim=100,final_dim=100)
        self.decoder = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 1000),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1000, 3072),
                nn.Tanh()
        )
 
    def forward(self, points):
        '''
        points: [bs,1024,3]
        '''
        encode = self.encoder(points.permute(0,2,1))  #[bs,100]
        rec_points = self.decoder(encode).view(-1,1024,3)
        return rec_points
    
    def encode(self, points):
        return self.encoder(points.permute(0,2,1))

class TAligNet(nn.Module):
    def __init__(self,args):
        super(TAligNet, self).__init__()
        self.autoencoders =  nn.ModuleList([AutoEncoder() for i in range(32)])
        self.MLP = nn.Sequential(
                nn.Linear(32*107, 1000),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1000, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 32*9),
                nn.Tanh()
        )
        # if args.checkpoint != "":
        #     checkpoint = torch.load(args.checkpoint)
        #     for i in range(32):
        #         self.autoencoders[i].load_state_dict(checkpoint,strict=True)
        self.norm = nn.BatchNorm1d(32*107)
        # self.norm = nn.LayerNorm(32*107)
        
        
    def forward(self,points,quaternion, centroid):
        '''
        points [bs,32,1024,3]
        '''
        bs = centroid.shape[0]
        n = centroid.shape[1]
        encodings = []
        for i in range(n):
            encoding = self.autoencoders[i].encode(points[:,i])
            encodings.append(encoding)
        encoding = torch.stack(encodings,dim=1)
        encoding = torch.cat([encoding,centroid,quaternion],dim=-1).view(bs,-1)
        # encoding = self.norm(encoding)
        dofs = self.MLP(encoding).view(bs,n,-1)
        return dofs