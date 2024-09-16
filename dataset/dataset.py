
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import copy
import open3d as o3d

def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1

def  randomize_mesh_orientation2(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    translation = np.array([random.uniform(-0.04,0.04),random.uniform(-0.04,0.04),random.uniform(-0.01,0.01)])
    centroid = mesh.centroid
    mesh1.vertices = rotation.apply(mesh1.vertices - centroid) + centroid + translation
    return mesh1, rotation, translation

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    return xyz

def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh

class TeethDataManager:
    def __init__(self, dataroot, train_ratio=0.9,augment=None):
        super().__init__()
        self.dataroot = Path(dataroot)
        self.augments = augment
        self.train_ratio = train_ratio
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.mesh_paths = []
        self.browse_dataroot()
    
    def browse_dataroot(self):
        for tooth in self.dataroot.iterdir():
            self.mesh_paths.append(tooth)
        
        random.shuffle(self.mesh_paths)
        split_point = int(len(self.mesh_paths) * self.train_ratio)
        self.train_objs = self.mesh_paths[:split_point]
        self.test_objs = self.mesh_paths[split_point:]
    
    def train_dataset(self):
        dataset = teethDataset(self.train_objs,True,self.augments)
        return dataset

    def test_dataset(self):
        dataset = teethDataset(self.test_objs,False,self.augments)
        return dataset

def load_mesh_shape(mesh, augments=[], request=[], seed=None):

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch

    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=int)), 0)
    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha

class teethDataset(data.Dataset):
    def __init__(self, objs, train=True, augment=None):
        super().__init__()

        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.mesh_paths = objs

        if train and augment:
            self.augments = augment
    
    def __len__(self):
        return len(self.mesh_paths)
    
    def __getitem__(self, idx):
        label = 0
        mesh = trimesh.load_mesh(self.mesh_paths[idx])
        feats, center, cordinates, faces, Fs = load_mesh_shape(mesh, augments=self.augments,
                                                             request=self.feats)

        return   feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx])


class FullTeethDataset(data.Dataset):
    def __init__(self, args, index_file, train=True):
        super().__init__()
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.dataroot = args.dataroot
        self.before_path = args.before_path
        self.after_path = args.after_path
        self.point_num = args.point_num
        if train:
            self.paramroot = args.paramroot
        with open(index_file) as f:
            self.indexes = [int(p.strip()) for p in f.readlines()]
        self.train = train
    
    def __getitem__(self, idx):
        point_num = self.point_num
        feats = np.zeros((32,10,256,64))
        center = np.zeros((32,256,64,3))
        cordinates = np.zeros((32,256,64,9))
        faces = np.zeros((32,256,64,3))
        Fs = np.zeros(32)
        before_points = np.zeros((32,point_num,3))
        after_points = np.zeros((32,point_num,3))
        centroid = np.zeros((32,3))
        after_centroid = np.zeros((32,3))
        quaternion = np.zeros((32,4))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        if self.train:
            rotation_matrix = torch.load(os.path.join(self.paramroot,f'{index}.pkl'))   #gt rotation matrix
        for i in range(32):
            obj_path = os.path.join(self.dataroot,f'{index}_{i}.obj')
            before_path = os.path.join(self.before_path,f'{index}_{i}.ply')
            after_path = os.path.join(self.before_path,f'{index}_{i}.ply')
            if os.path.exists(obj_path) and os.path.exists(before_path) and os.path.exists(after_path):
                mesh = trimesh.load_mesh(obj_path, process=False)
                masks[i] = 1
                before = read_pointcloud(before_path)
                before_points[i] = before[:point_num]
                centroid[i] = before[point_num]
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num]
                after_centroid[i] = after[point_num]
                if self.train and np.random.rand() < 0.3:
                    mesh, rotation, translation = randomize_mesh_orientation2(mesh)
                    before_points[i] = rotation.apply(before_points[i] - np.expand_dims(centroid[i],0)) + np.expand_dims(centroid[i] + translation,0)
                    centroid[i] = centroid[i] + translation
                    rotation_matrix[i] = np.matmul(rotation.as_matrix(),rotation_matrix[i])
                try:
                    feats[i], center[i], cordinates[i], faces[i], Fs[i]= load_mesh_shape(mesh, 
                                                                    request=self.feats)
                except:
                    continue
                obb = mesh.bounding_box_oriented
                rotation = copy.copy(obb.primitive.transform[:3, :3])
                rotation = Rotation.from_matrix(rotation)
                quaternion[i] = rotation.as_quat()
        if self.train:
            rotation_matrix = rotation_matrix.reshape(32,9)[:,:6]
            gt_params = torch.cat([torch.from_numpy(after_centroid.copy()),rotation_matrix],dim=-1)
            return   feats,center,cordinates,faces,Fs,index,before_points,after_points,centroid,after_centroid,gt_params,masks,quaternion
        else:
            return   feats,center,cordinates,faces,Fs,index,before_points,after_points,centroid,after_centroid,masks,quaternion


    def __len__(self):
        return len(self.indexes)
