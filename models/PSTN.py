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
from models.pointnet import PointNet, PointNetEncoder, PointNetPlusEncoder
import torch

class PSTN(nn.Module):
    def __init__(self,args):
        super(PSTN, self).__init__()
        # self.p1_local_encoder = PointNet(final_dim=1024)
        self.p1_local_encoders = nn.ModuleList([PointNet(final_dim=1024) for i in range(32)])
        self.p1_global_encoder = PointNet(final_dim=1024)
        self.p2_global_encoder = PointNetPlusEncoder(128*32,final_dim=512)
        # self.U = PointNet(channel=2563,first_dim=1024,second_dim=512,final_dim=512)
        self.regressor = nn.Sequential(
                nn.Linear(2947, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 32*9),
                nn.Tanh()
        )
        self.bn = nn.BatchNorm1d(2563)
        
        
    def forward(self,faces, centroid, points):
        '''
        points [bs,32,128,3]
        '''
        bs = faces.shape[0]
        n = faces.shape[1]
        encodings = []
        for i in range(n):
            encoding = self.p1_local_encoders[i](points[:,i].permute(0,2,1))
            # encoding = self.p1_local_encoder(points[:,i].permute(0,2,1))
            encodings.append(encoding)
        p1_local_embedding = torch.stack(encodings,dim=1)  #[bs,32,1024]

        global_points = points.view(bs,-1,3)
        p1_global_embedding = self.p1_global_encoder(global_points.permute(0,2,1)).unsqueeze(1).repeat(1,n,1)

        p2_global_embedding = self.p2_global_encoder(global_points.permute(0,2,1)).unsqueeze(1).repeat(1,n,1)
        embedding = torch.cat([points.view(bs,n,-1),p1_local_embedding,p1_global_embedding,p2_global_embedding,centroid],dim=-1) #[bs,32,2947]
        # embedding = torch.cat([p1_local_embedding,p1_global_embedding,p2_global_embedding,centroid],dim=-1)
        # parameter = self.U(embedding.transpose(1,2))
        dofs = self.regressor(embedding).reshape(bs,n,-1)
        return dofs