# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 32 # 16 or 32
residual_blocks=True #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2

import torch, sod_data, iou
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np
from plyfile import PlyData, PlyElement
from prepare_data import prepare_back_remapper

output_dir = './sod_scn_vis'


def add_labels_to_ply(plydata, prop_name, prop_type, prop_value):
    v = plydata.elements[0]
    f = plydata.elements[1]

    # Create the new vertex data with appropriate dtype
    if prop_name not in v.data.dtype.fields:
        a = np.empty(len(v.data), v.data.dtype.descr +
                     [(prop_name, prop_type)])
    else:
        a = np.empty(len(v.data), v.data.dtype.descr)
    for name in v.data.dtype.fields:
        a[name] = v[name]
    a[prop_name] = prop_value

    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v, f], text=False)
    return p



# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]


def add_colors_from_labels(plydata, colors, labels):
    for i in range(0, len(labels)):
        color = colors[labels[i]]
        plydata['vertex']['red'][i] = color[0]
        plydata['vertex']['green'][i] = color[1]
        plydata['vertex']['blue'][i] = color[2]


def read_ply(ply_path):
    verts = None
    labels = None
    instances = None
    faces = None
    normals = None
    r = None
    g = None
    b = None
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)
        for el in plydata.elements:
            if el.name == "vertex":
                verts = np.stack(
                    [el.data["x"], el.data["y"], el.data["z"]], axis=1)

                # if 'Nx' in el.data:
                # normals = np.stack(
                # [el.data["Nx"], el.data["Ny"], el.data["Nz"]], axis=1)
                is_label = False
                is_instance = False
                for pr in el.properties:
                    if pr._name == "label":
                        is_label = True
                    if pr._name == "instance":
                        is_instance = True
                    if pr._name == "red":
                        r = el.data["red"]
                    if pr._name == "green":
                        g = el.data["green"]
                    if pr._name == "blue":
                        b = el.data["blue"]
                if is_label:
                    labels = el.data["label"]
                if is_instance:
                    instances = el.data["instance"]
            if el.name == "face":
                flist = []
                for f in el.data:
                    flist.append(f[0])
                faces = np.asarray(flist)
    rgb = None
    if r is not None:
        rgb = np.stack([r, g, b], axis=1)
    return verts, labels, instances, faces, normals, rgb, plydata


def load_labelled_sod_mesh(scene_name):
    mesh_path = f'/data/data/SOD_ScanNet_Format/sod_single_with_gt_meshes/{scene_name}/global_map_mesh.clean.labels.ply'
    verts, labels, instances, faces, normals, rgb, plydata = read_ply(mesh_path)
    return verts, plydata


use_cuda = torch.cuda.is_available()
exp_name='unet_scale20_m16_rep1_notResidualBlocks'

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(sod_data.dimension,sod_data.full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(sod_data.dimension, 3, m, 3, False)).add(
               scn.UNet(sod_data.dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(sod_data.dimension))
        self.linear = nn.Linear(m, 20)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x

unet=Model()
if use_cuda:
    unet=unet.cuda()

training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

back_remapper = prepare_back_remapper()
colors = create_color_palette()

with torch.no_grad():
    unet.eval()
    store=torch.zeros(sod_data.sodOffsets[-1],20)
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    for rep in range(1,1+sod_data.val_reps):
        for i,batch in enumerate(sod_data.sod_data_loader):
            if use_cuda:
                batch['x'][1]=batch['x'][1].cuda().float()
                batch['y']=batch['y'].cuda()
            predictions = unet(batch['x'])
            store.index_add_(0,batch['point_ids'],predictions.cpu())
            sid = batch['id'][0]
            print(sid)
            file_name = sod_data.sod_files[sid]
            scene_name = file_name[4:-4]
            print(scene_name)
            mesh_point_ids = batch['point_ids'] - sod_data.sodOffsets[sid]

            verts, plydata = load_labelled_sod_mesh(scene_name)
            #set labels
            labels = predictions.cpu().max(1)[1].numpy()                
            labels_scannet = np.zeros((verts.shape[0]), dtype=np.int32)
            labels_scannet_predicted = back_remapper[labels]
            labels_scannet[mesh_point_ids] = labels_scannet_predicted
            plydata = add_labels_to_ply(plydata, 'label', 'i4', labels_scannet)
            #add colors
            add_colors_from_labels(plydata, colors, labels_scannet)
            #save mesh
            plydata.write(f'{output_dir}/{scene_name}.ply')

        print('SOD MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(sod_data.sod)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(sod_data.sod)/1e6,'time=',time.time() - start,'s')
        print(store.shape)
        m = store.max(1)[1]       
        print(m) 
        iou.evaluate(m.numpy(),sod_data.sodLabels)

        

