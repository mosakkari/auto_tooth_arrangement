import torch
import os
import sys
from torch.autograd import Variable
import argparse
from collections import OrderedDict
# from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from models.origin import Origin
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from models.TAligNet import TAligNet
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from sklearn.manifold import TSNE
from models.PSTN import PSTN
import matplotlib
import vedo
from pytorch3d.transforms import Transform3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import trimesh
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation
from einops import rearrange
from scipy.interpolate import splprep, splev
from scipy.interpolate import make_interp_spline
from pytorch3d.transforms import se3_exp_map,se3_log_map
from pytorch3d.transforms import euler_angles_to_matrix, rotation_6d_to_matrix
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_quaternion,quaternion_to_axis_angle
from scipy.spatial.transform import Rotation
from dataset.dataset import FullTeethDataset
from models.tadpm import TADPM
from models.tanet import TANet
from frechetdist import frdist

def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = torch.nn.functional.normalize(x_raw,dim=-1)  # batch*3
    y = torch.nn.functional.normalize(y_raw,dim=-1)  # batch*3
    middle = torch.nn.functional.normalize(x + y,dim=-1)
    orthmid = torch.nn.functional.normalize(x - y,dim=-1)
    x = torch.nn.functional.normalize(middle + orthmid,dim=-1)
    y = torch.nn.functional.normalize(middle - orthmid,dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y, dim=-1),dim=-1)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def move_mesh(mesh,centroid,dof):
    move = (dof[:3]).unsqueeze(0) #[b*n,1,3]
    vertices = torch.FloatTensor(mesh.vertices).cuda()
    vertices = vertices + move
    mesh.vertices = vertices.cpu().numpy()
    return mesh

def transform_teeth(index,centers,RR,args):
    before_meshes = []
    meshes = []
    gt_meshes = []
    before_meshes_upper = []
    before_meshes_lower = []
    meshes_upper = []
    meshes_lower = []
    gt_meshes_upper = []
    gt_meshes_lower = []
    valid = 0
    for i in range(32):
        path = os.path.join(args.before_mesh_path,f'{index}_{i}.obj')
        gt_path = os.path.join(args.after_mesh_path,f'{index}_{i}.obj')
        if os.path.exists(path) and os.path.exists(gt_path):
            valid += 1
            mesh = trimesh.load_mesh(path)
            gt_mesh = trimesh.load_mesh(gt_path)
            before_mesh = trimesh.load_mesh(path)
            mesh.vertices = np.matmul((mesh.vertices - mesh.centroid),RR[i].cpu().numpy()) + centers[i].cpu().numpy()
            if args.segmented:
                if i<16:
                    before_meshes_upper.append(vedo.trimesh2vedo(before_mesh))
                    meshes_upper.append(vedo.trimesh2vedo(mesh))
                    gt_meshes_upper.append(vedo.trimesh2vedo(gt_mesh))
                else:
                    before_meshes_lower.append(vedo.trimesh2vedo(before_mesh))
                    meshes_lower.append(vedo.trimesh2vedo(mesh))
                    gt_meshes_lower.append(vedo.trimesh2vedo(gt_mesh))
            else:
                before_meshes.append(vedo.trimesh2vedo(before_mesh))
                meshes.append(vedo.trimesh2vedo(mesh))
                gt_meshes.append(vedo.trimesh2vedo(gt_mesh))
    if args.segmented:
        before_mesh_upper = vedo.merge(before_meshes_upper)
        before_mesh_lower = vedo.merge(before_meshes_lower)
        mesh_upper = vedo.merge(meshes_upper)
        mesh_lower = vedo.merge(meshes_lower)
        gt_mesh_upper = vedo.merge(gt_meshes_upper)
        gt_mesh_lower = vedo.merge(gt_meshes_lower)
        outputroot = args.outputroot
        os.makedirs(outputroot,exist_ok=True)
        new_index = index
        vedo.write(before_mesh_upper,os.path.join(outputroot,f'{new_index}_upper_before.stl'))
        vedo.write(mesh_upper,os.path.join(outputroot,f'{new_index}_upper_after.stl'))
        vedo.write(gt_mesh_upper,os.path.join(outputroot,f'{new_index}_upper_gt.stl'))
        vedo.write(before_mesh_lower,os.path.join(outputroot,f'{new_index}_lower_before.stl'))
        vedo.write(mesh_lower,os.path.join(outputroot,f'{new_index}_lower_after.stl'))
        vedo.write(gt_mesh_lower,os.path.join(outputroot,f'{new_index}_lower_gt.stl'))
    else:
        before_mesh = vedo.merge(before_meshes)
        mesh = vedo.merge(meshes)
        gt_mesh = vedo.merge(gt_meshes)
        outputroot = args.outputroot
        os.makedirs(outputroot,exist_ok=True)
        new_index = index
        vedo.write(before_mesh,os.path.join(outputroot,f'{new_index}_before.stl'))
        vedo.write(mesh,os.path.join(outputroot,f'{new_index}_after.stl'))
        vedo.write(gt_mesh,os.path.join(outputroot,f'{new_index}_gt.stl'))

def test(net, test_dataset, args):

    net.eval()

    n_samples = 0
    sum = 0
    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid, masks,quaternion) in enumerate(
            test_dataset):
        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        quaternion = quaternion.to(torch.float32).cuda()
        masks = masks.cuda()
        n_samples += faces.shape[0]
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32).cuda()
            predicted_centroid = outputs[:,:,:3]
            dofs = rearrange(outputs[:,:,3:],'b n c -> (b n) c')
            rot_matrix = rotation_6d_to_matrix(dofs)
            rot_matrix = rearrange(rot_matrix,'(b n) c1 c2 -> b n c1 c2', n=32)
            for i in range(index.shape[0]):
                transform_teeth(index[i],predicted_centroid[i],rot_matrix[i],args)

def cal_average(file_name):
    with open(file_name) as f:
        lines = [float(i.strip()) for i in f.readlines()]
    val = torch.tensor(lines).mean()
    os.system(f'rm {file_name}')
    return val.item()
            

if __name__ == '__main__':
    seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr_milestones', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=str, default=None)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, required=True, default=500)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--segmented', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--point_num', type=int, default=512)
    parser.add_argument('--encoder_checkpoint',type=str,default='')
    parser.add_argument('--before_path',type=str, required=True)
    parser.add_argument('--after_path',type=str, required=True)
    parser.add_argument('--before_mesh_path',type=str, required=True)
    parser.add_argument('--after_mesh_path',type=str, required=True)
    parser.add_argument('--outputroot',type=str, required=True)
    parser.add_argument('--checkpoint',type=str,default='')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    args = parser.parse_args()
    # mode = args.mode
    args.name = args.name
    dataroot = args.dataroot

    # ========== Dataset ==========
    augments = []
    test_dataset = FullTeethDataset(args,'test.txt',False)
    print(len(test_dataset))
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TADPM(args).to(device)
    net = nn.DataParallel(net)
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        print('loading model...')
        net.load_state_dict(checkpoint['model'],strict=True)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    

    # ========== Start Training ==========
    test(net, test_data_loader, args)


