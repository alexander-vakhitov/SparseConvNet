from prepare_data import prepare_scene, prepare_remapper
import torch

if __name__ == '__main__':
    remapper = prepare_remapper()
    sod_path = '/data/data/SOD_ScanNet_Format/sod_single_with_gt_meshes/'
    for scene in ['big_table1', 'kitchen1', 'lab1', 'small_meeting_room1']:
        fn = f'{sod_path}/{scene}/global_map_mesh.clean.ply'
        fn2 = f'{sod_path}/{scene}/global_map_mesh.clean.labels.ply'
        (coords,colors,w) = prepare_scene(fn, fn2, remapper)
        torch.save((coords,colors,w),f'./sod/{scene}.pth')