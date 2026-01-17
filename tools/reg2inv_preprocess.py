#!/usr/bin/env python3
"""
使用训练好的 Reg2Inv 模型对点云进行离线配准的工具脚本。

用法示例：
python tools/reg2inv_preprocess.py \
  --input_dir data/mvtec3d/carrot/xyz \
  --output_dir data/mvtec3d/carrot/xyz_registered \
  --reg2inv_root Reg2Inv-main \
  --checkpoint output_mvtec3d/snapshots/latest.pth.tar \
  --device cuda
python tools/reg2inv_preprocess.py \
  --input_dir data/mvtec3d/ \
  --output_dir data/mvtec3d/ \
  --template_path data/mvtec3d/carrot/train/template.pcd \
  --checkpoint output_mvtec_carrot/snapshots/snapshot.pth.tar \
  --device cuda

说明：脚本会尝试加载 Reg2Inv 的模型并调用其前向推理得到 output_dict['estimated_transform']。
如果模型不可用或推理失败，会回退到 PCA 对齐（并可扩展为 PCA+ICP）。
"""

#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import numpy as np
import open3d as o3d
import torch

# --- 1. 动态添加项目根目录到环境变量，确保能 import 项目模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 尝试导入项目模块
try:
    from config_mvtec3d import make_cfg
    from dataset_mvtec3d import train_valid_data_loader
    from model import create_model
    from geotransformer.utils.data import precompute_data_stack_mode_two
    from geotransformer.utils.torch import to_cuda
except ImportError as e:
    print(f"Error: 无法导入项目模块。请确保脚本位于 Reg2Inv-main/tools/ 目录下。\n详细错误: {e}")
    sys.exit(1)


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    # 移除无效点 (NaN/Inf)
    pcd.remove_non_finite_points()
    return np.asarray(pcd.points, dtype=np.float32), pcd

def save_pcd(points, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 保存为 ASCII 格式方便查看，也可以设为 False 保存二进制
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)

def pca_align(points):
    """
    PCA 回退方案：将点云的主轴对齐到坐标轴，并将中心移到原点。
    """
    centroid = points.mean(axis=0)
    centered_pts = points - centroid
    U, S, Vh = np.linalg.svd(np.dot(centered_pts.T, centered_pts))
    
    # 旋转矩阵 R = U.T
    R = U.T
    
    # 构建变换矩阵 T (4x4)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ centroid
    return T

class Reg2InvInfer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        self.model = None
        self.cfg = make_cfg()
        
        print(f"[Init] Loading config and calibrating neighbors...")
        # 我们利用 train_loader 来获取正确的 neighbor_limits 和 voxel_size
        # 这里的 dataset_root 只是为了跑通过程，实际推理不依赖它
        # 注意：如果你的 config 路径写死了，这里可能会去读一下数据
        try:
            _, _, neighbor_limits, voxel_size = train_valid_data_loader(self.cfg, distributed=False)
        except Exception as e:
            print(f"[Warning] DataLoader 初始化失败，尝试使用默认参数。错误: {e}")
            # 如果失败，给一组经验值 (针对 MVTec)
            neighbor_limits = [38, 36, 36] 
            voxel_size = 0.03

        self.voxel_size = voxel_size
        self.neighbor_limits = neighbor_limits
        
        print(f"[Init] Creating model (Voxel: {self.voxel_size}, Neighbors: {self.neighbor_limits})...")
        self.model = create_model(self.cfg, self.voxel_size, self.neighbor_limits)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[Init] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理 state_dict 可能包含的 'module.' 前缀
            state_dict = checkpoint.get('model', checkpoint)
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v # 去掉 module.
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
        else:
            print(f"[Error] Checkpoint not found at {checkpoint_path}")
            self.model = None

    def infer(self, ref_points, src_points):
        """
        计算 src 到 ref 的变换矩阵
        """
        if self.model is None:
            return None

        # 准备数据 input_dict
        ref_len = [ref_points.shape[0]]
        src_len = [src_points.shape[0]]
        
        # 预计算多尺度数据
        input_dict = precompute_data_stack_mode_two(
            ref_points, src_points, ref_len, src_len,
            self.cfg.backbone.num_stages,
            self.voxel_size,
            self.cfg.backbone.init_radius,
            self.neighbor_limits
        )
        
        input_dict = to_cuda(input_dict)
        
        try:
            with torch.no_grad():
                output_dict = self.model(input_dict)
            
            est_transform = output_dict['estimated_transform'] # (1, 4, 4)
            return est_transform.squeeze(0).cpu().numpy()
            
        except Exception as e:
            print(f"[Inference Error] {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Reg2Inv Offline Registration Tool")
    parser.add_argument('--input_dir', type=str, required=True, help='待配准点云目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--template_path', type=str, required=True, help='参考模版点云路径 (Template)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth.tar)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 1. 初始化模型
    inferencer = Reg2InvInfer(args.checkpoint, args.device)
    
    # 2. 加载模版 (Reference)
    if not os.path.exists(args.template_path):
        print(f"Error: Template file not found: {args.template_path}")
        return
    print(f"Loading Template: {args.template_path}")
    ref_points, _ = load_pcd(args.template_path)

    # 3. 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 遍历输入文件
    # 支持多种后缀
    exts = ['*.pcd', '*.ply', '*.xyz', '*.txt']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    files.sort()
    
    print(f"Found {len(files)} files to process.")

    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        save_path = os.path.join(args.output_dir, filename)
        
        print(f"[{i+1}/{len(files)}] Processing {filename} ...", end=' ')
        
        # 加载源点云
        src_points, _ = load_pcd(file_path)
        
        # 推理
        transform_matrix = inferencer.infer(ref_points, src_points)
        
        final_points = src_points
        
        if transform_matrix is not None:
            # 应用 Reg2Inv 变换
            # Points (N, 3) @ T[:3,:3].T + T[:3,3]
            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]
            final_points = np.dot(src_points, R.T) + t
            print("Done (Reg2Inv).")
        else:
            # 回退到 PCA
            print("Failed -> Fallback (PCA).")
            T_pca = pca_align(src_points)
            R = T_pca[:3, :3]
            t = T_pca[:3, 3]
            final_points = np.dot(src_points, R.T) + t

        # 保存结果
        save_pcd(final_points, save_path)

    print(f"\nAll processed. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()