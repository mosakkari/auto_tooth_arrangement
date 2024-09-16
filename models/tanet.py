import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from chamfer_dist import ChamferDistanceL1
import copy
import numpy as np
from models.GGNN import GGNN
import math
from functools import partial
from models.pointnet import PointNet
import torch

class TANet(nn.Module):
    def __init__(self,args):
        super(TANet, self).__init__()
        # self.encoder = PointNet(final_dim=512)
        self.encoders = nn.ModuleList([PointNet(final_dim=512) for _ in range(32)])

        self.regressor = nn.Sequential(
            nn.Linear(1664, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 9),
            nn.Tanh()
        )
        embed_dim = args.dim
        self.embed_dim = embed_dim
        self.global_encoder = PointNet(final_dim = 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.norm_layer = nn.LayerNorm(512)
        self.propagation_model = GGNN()
        A = torch.zeros(34,34*8).cuda()
        for i in range(32):  
            if i!=15 and i!=31: 
                A[i,i+1] = 1
                A[i+1,i] = 1
            if i<=15:   
                A[i,34+15-i] = 1
                A[15-i,34+i] = 1

                A[i,68+32] = 1
                A[32,68+i] = 1
            else:
                A[i,34+47-i] = 1
                A[47-i,34+i] = 1

                A[i,68+33] = 1
                A[33,68+i] = 1
        A[32,102+33] = 1
        A[33,102+32] = 1
        A[:,136:] = A[:,:136]
        self.A = nn.parameter.Parameter(A,requires_grad=False)
        
        
    def forward(self,faces, centroid, points, epsilon):
        '''
        points [bs,16,2048,3]
        '''
        bs = faces.shape[0]
        n = faces.shape[1]
        encodings = []
        centered_points = points - centroid.unsqueeze(2).repeat(1,1,points.shape[2],1)
        for i in range(n):
            # encoding = self.encoder(centered_points[:,i].permute(0,2,1))
            encoding = self.encoders[i](centered_points[:,i].permute(0,2,1))
            encodings.append(encoding)
        for _ in range(2):
            encodings.append(torch.zeros_like(encodings[0]))
        embedding = torch.stack(encodings,dim=1)  #[bs,16,768]
        add_feature = self.propagation_model(embedding,self.A.unsqueeze(0).repeat(bs,1,1).to(embedding.device))
        trans_feature = embedding[:,:32] + add_feature[:,:32]
        trans_feature = self.norm_layer(trans_feature)
        global_embedding = self.global_encoder(points.view(bs,-1,3).permute(0,2,1)).unsqueeze(1).repeat(1,n,1)
        center_emb = centroid.view(faces.shape[0],-1).unsqueeze(1).repeat(1,n,1)
        embedding = torch.cat([global_embedding,trans_feature,center_emb,epsilon],dim=-1)
        dofs = self.regressor(embedding)
        return dofs