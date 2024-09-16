import torch
import os
import sys
from torch.autograd import Variable
import argparse
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from sklearn.manifold import TSNE
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
from einops import rearrange, repeat
from pytorch3d.transforms import rotation_6d_to_matrix
from dataset.dataset import FullTeethDataset
from models.tadpm import TADPM
from models.util import progress_bar

def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_matrix(centroid):
    return torch.cdist(centroid,centroid)

def get_matrix_mask(masks):
    f_masks = masks.float()
    matrix_mask = torch.bmm(f_masks.unsqueeze(-1),f_masks.unsqueeze(-2))
    return matrix_mask


def train(net, optim, names, scheduler, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid,gt_params,masks, quaterion) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        gt_params = gt_params.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.cuda()
        centroid = centroid.to(torch.float32).cuda()
        after_centroid = after_centroid.to(torch.float32).cuda()
        masks = masks.cuda()
        n_samples += faces.shape[0]
        before_points = before_points.to(torch.float32).cuda()
        after_points = after_points.to(torch.float32).cuda()
        outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points, gt_params).to(torch.float32).cuda()
        predicted_centroid = outputs[:,:,:3]
        dofs = rearrange(outputs[:,:,3:],'b n c -> (b n) c')
        criterion = nn.MSELoss(reduction='none')
        rot_matrix = rotation_6d_to_matrix(dofs)
        predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
        predicted_points = torch.bmm(predicted_points,rot_matrix)
        predicted_points = predicted_points + rearrange(predicted_centroid,'b n c->(b n) c').unsqueeze(1)
        after_points = rearrange(after_points,'b n p c -> (b n) p c')

        # If the points before and after are not in one-to-one correspondence, use chamfer distance instead
        loss1 = criterion(predicted_points,after_points).sum(dim=(1,2))
        loss1 = 0.001*(loss1 * masks.flatten()).sum()

        # loss for diffusion
        loss2 = 0.03*((criterion(outputs,gt_params).sum(dim=-1)) * masks).sum()

        # position loss
        loss3 = 0.1*(criterion(get_matrix(centroid), get_matrix(predicted_centroid)) * get_matrix_mask(masks)).sum()
        loss = loss1 + loss2 + loss3
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)
        progress_bar(it, len(train_dataset), 'Train Loss: %.3f'% (running_loss/n_samples))

    scheduler.step()
    epoch_loss = running_loss / n_samples
    message = 'epoch ({:}): {:} Train Loss: {:.4f}'.format(names, epoch, epoch_loss)
    with open(os.path.join(args.saveroot, names, 'log.txt'), 'a') as f:
        f.write(message+'\n')
    print()
    print(message)
    if (epoch+1)%50 == 0:
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save({'model':best_model_wts,'optim':optim.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch}, os.path.join(args.saveroot, names, f'acc-{epoch_loss:.4f}-{epoch}.pkl'))

def test(net, names, optimizer, scheduler, test_dataset, epoch, args, autoencoder=None):

    net.eval()
    if autoencoder!=None:
        autoencoder.eval()

    running_loss = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, index, before_points, after_points, centroid,after_centroid,masks, quaterion) in enumerate(
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
        masks = masks.cuda()
        n_samples += faces.shape[0]
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates, centroid, before_points).to(torch.float32).cuda()
            predicted_centroid = outputs[:,:,:3]
            dofs = rearrange(outputs[:,:,3:],'b n c -> (b n) c')
            criterion = nn.MSELoss(reduction='none')
            rot_matrix = rotation_6d_to_matrix(dofs)
            predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
            predicted_points = torch.bmm(predicted_points,rot_matrix)
            predicted_points = predicted_points + rearrange(predicted_centroid,'b n c->(b n) c').unsqueeze(1)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            loss = criterion(predicted_points,after_points).sum(dim=(1,2))
            loss = 0.001*(loss * masks.flatten()).sum()
            running_loss += loss.item() * faces.size(0)
            progress_bar(it, len(test_dataset), 'Test Loss: %.3f'% (running_loss/n_samples))

    epoch_loss = running_loss / n_samples
    if test.best_loss > epoch_loss:
        test.best_loss = epoch_loss
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save({'model':best_model_wts,'optim':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch}, os.path.join(args.saveroot, names, 'best_acc.pkl'))
        # torch.save(best_model_wts, os.path.join('/data/lcs/created_checkpoints', names, 'best_acc.pkl'))

    message = 'epoch ({:}): {:} test Loss: {:.4f}'.format(names, epoch, epoch_loss)
    with open(os.path.join(args.saveroot, names, 'log.txt'), 'a') as f:
        f.write(message+'\n')
    print()
    print(message)

if __name__ == '__main__':
    seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
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
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--saveroot', type=str, required=True)
    parser.add_argument('--paramroot', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--before_path',type=str, required=True)
    parser.add_argument('--after_path',type=str, required=True)
    parser.add_argument('--encoder_checkpoint',type=str,default='')
    parser.add_argument('--checkpoint',type=str,default='')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--point_num', type=int, default=512)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    args = parser.parse_args()
    mode = args.mode
    args.name = args.name
    dataroot = args.dataroot
    paramroot = args.paramroot

    # ========== Dataset ==========
    augments = []
    train_dataset = FullTeethDataset(args,'train.txt',True)
    test_dataset = FullTeethDataset(args,'test.txt',False)
    print(len(train_dataset))
    print(len(test_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True, drop_last=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TADPM(args).to(device)
    net = nn.DataParallel(net)
    if args.checkpoint != '':
        print('...loading...')
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model'],strict=True)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # ========== Optimizer ==========
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epoch)
    checkpoint_names = []
    checkpoint_path = os.path.join(args.saveroot, args.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    train.step = 0
    test.best_loss = 99999

    # ========== Start Training ==========
    for epoch in range(args.n_epoch):
        print('epoch', epoch)
        train(net, optimizer, args.name, scheduler, train_data_loader, epoch, args)
        print('train finished')
        test(net, args.name, optimizer, scheduler, test_data_loader, epoch, args)
        print('test finished')
        