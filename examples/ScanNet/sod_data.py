# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
scale=50  #Voxel size = 1/scale
val_reps=1 # Number of test views, 1 or more
batch_size=1
elastic_deformation=False

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time

dimension=3
full_scale=4096 #Input field size

# Class IDs have been mapped to the range {0,1,...,19}
# NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

num_dataloader_workers = 4

sod = []
sod_files = glob.glob('sod/*.pth')

for x in torch.utils.data.DataLoader(
        sod_files,
        collate_fn=lambda x: torch.load(x[0]), num_workers=num_dataloader_workers):
    sod.append(x)
print('SOD examples:', len(sod))
print('SOD files:')
print(sod_files)


sodOffsets = [0]
sodLabels = []
for idx,x in enumerate(sod):
    sodOffsets.append(sodOffsets[-1]+x[2].size)
    sodLabels.append(x[2].astype(np.int32))
sodLabels = np.hstack(sodLabels)

def sodMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    point_ids=[]
    for idx,i in enumerate(tbl):
        a,b,c = sod[i]
        m=np.eye(3)
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*math.pi
        m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        a=np.matmul(a,m)+full_scale/2+np.random.uniform(-2,2,3)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        point_ids.append(torch.from_numpy(np.nonzero(idxs)[0]+sodOffsets[i]))        
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    point_ids=torch.cat(point_ids,0)
    return {'x': [locs,feats], 'y': labels.long(), 'id': tbl, 'point_ids': point_ids}
sod_data_loader = torch.utils.data.DataLoader(
    list(range(len(sod))),
    batch_size=batch_size,
    collate_fn=sodMerge,
    num_workers=num_dataloader_workers,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
