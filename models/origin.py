import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from collections import OrderedDict
from models.diffusion import diffuse
import numpy as np
import math
from functools import partial
from models.mesh_mae import Mesh_encoder
import torch
import timm.models.vision_transformer
from models.pointnet import PointNetEncoder
from timm.models.vision_transformer import PatchEmbed, Block
class Origin(nn.Module):
    def __init__(self,args):
        super(Origin, self).__init__()
        
    def forward(self,centroid, points):
        '''
        points [bs,32,2048,3]
        '''
        bs = centroid.shape[0]
        rot = torch.tensor([[[1,0,0,0,1,0]]]).repeat(bs,32,1).cuda()
        dofs = torch.cat([centroid,rot],dim=-1)
        # dofs = self.regressor(embedding)
        return dofs

 