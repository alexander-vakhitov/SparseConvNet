import os, shutil

if __name__ == '__main__':
    scannet_scene_lst_path = '/home/sasha/differentiable_slam_map/configs/scannetv2_train.txt'
    scannet_root_path = '/mnt/data/ScanNet/scans'
    output_path = "/home/sasha/SparseConvNet/examples/ScanNet/train/"

#    scannet_scene_lst_path = '/home/alexander/projects/differentiable_slam_map/configs/scannetv2_val.txt'
    # scannet_root_path = '/data/data/scannet/scans/'
    
#    output_path = "/home/alexander/soft/SparseConvNet/examples/ScanNet/val/"

    with open(scannet_scene_lst_path) as f_in:
        for line in f_in:
            scene_id = line[0:-1]
            for mesh_path in [f"{scannet_root_path}/{scene_id}/{scene_id}_vh_clean_2.labels.ply", f"{scannet_root_path}/{scene_id}/{scene_id}_vh_clean_2.ply"]:
                if not os.path.exists(mesh_path):
                    print('Error! ' + mesh_path)
                    exit()
                shutil.copy(mesh_path, output_path)
