import contextlib
import logging
import os
import sys
import click
import numpy as np
import torch
import tqdm
import patchcore.sampler
from utils.utils import set_torch_device,fix_seeds
import pandas as pd
from dataset_mvtec3d import test_data_loader
from config_mvtec3d import make_cfg
from geotransformer.utils.torch import to_cuda
from model import create_model
from geotransformer.modules.ops import apply_transform
import open3d as o3d
LOGGER = logging.getLogger(__name__)

@click.command()
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--snapshots", default='output_mvtec3d/', required=True, help="Path to the snapshot directory") # 设为必填

def main(gpu, snapshots):
    device = set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    
    cfg = make_cfg()
    
    # For building PatchCore memory bank we use bank_data_loader
    test_loader, neighbor_limits, voxel_size  = test_data_loader(cfg)
    with device_context:
        torch.cuda.empty_cache()
        snapshot = snapshots + 'snapshots/snapshot.pth.tar'
        deep_feature_extractor = create_model(cfg, voxel_size, neighbor_limits).cuda()
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        deep_feature_extractor.load_state_dict(state_dict['model'], strict=True)
        deep_feature_extractor.eval()

        total_iterations = len(test_loader)
        pbar = tqdm.tqdm(enumerate(test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
         
            with torch.no_grad():
                data_dict = to_cuda(data_dict)
                output_dict = deep_feature_extractor(data_dict)
                input_pointcloud = data_dict['raw_points']
                # point_cloud = input_pointcloud.cpu()
                est_transform = output_dict['estimated_transform']
                # src_points = output_dict['src_points_f']        
                est_raw_points = apply_transform(input_pointcloud, est_transform)
                est_raw_points_np = est_raw_points.cpu().numpy()  # Shape of est_raw_points_np: (46087, 3)
                orig_path = data_dict['path']
                # 原始: .../class/test/defect/pcd/000.pcd
                # 目标: .../class/test/defect/register_pcd/000.pcd
                dir_name = os.path.dirname(orig_path) # .../class/test/defect/pcd
                file_name = os.path.basename(orig_path) # 000.pcd
                save_dir = dir_name.replace('pcd', 'register_pcd')
                print(f"Saving registered point cloud to: {os.path.join(save_dir, file_name)}")
                os.makedirs(save_dir, exist_ok=True)
                save_full_path = os.path.join(save_dir, file_name)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(est_raw_points_np.astype(np.float64))
                o3d.io.write_point_cloud(save_full_path, pcd, write_ascii=True)  # 保存 (ASCII 格式方便查看，如需更小体积可设 write_ascii=False)
                
    print("\n[Success] All registered point clouds have been saved.")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    main()