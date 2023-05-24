# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch

def prepare_scene(fn, fn2, remapper):
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    a=plyfile.PlyData().read(fn2)
    w=remapper[np.array(a.elements[0]['label'])]
    return (coords,colors,w)


# Map relevant classes to {0,1,...,19}, and ignored classes to -100
def prepare_remapper():
    remapper=np.ones(150)*(-100)
    for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
        remapper[x]=i
    return remapper


def prepare_back_remapper():
    back_remapper = np.ones(20)
    for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
        back_remapper[i] = x
    return back_remapper


if __name__ == '__main__':
    
    remapper = prepare_remapper()

    files=sorted(glob.glob('*/*_vh_clean_2.ply'))
    files2=sorted(glob.glob('*/*_vh_clean_2.labels.ply'))
    assert len(files) == len(files2)

    def f(fn):
        fn2 = fn[:-3]+'labels.ply'
        (coords,colors,w) = prepare_scene(fn, fn2, remapper)
        torch.save((coords,colors,w),fn[:-4]+'.pth')
        print(fn, fn2)

    p = mp.Pool(processes=mp.cpu_count())
    p.map(f,files)
    p.close()
    p.join()
