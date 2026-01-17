"""
随机旋转实验脚本

用法示例：
python multi_view/random_rotation_experiment.py \
  --data_path ./data/visa --rotations_per_sample 3 --mode full --out_suffix rot_full

说明：脚本不会执行 PointAD 推理，只在渲染管线调用前对点云应用旋转并保存渲染结果（2D 渲染图、GT 渲染图、3D->2D 对应）。
需要在同目录下能导入 `multiview_mvtec.get_mv_images`。
"""

import os
import sys
import argparse
import math
import numpy as np
from pathlib import Path

# 确保可以导入同目录下的模块
sys.path.append(os.path.dirname(__file__))
from multiview_mvtec import get_mv_images


def random_rotation_matrix_full():
    # 基于 Shoemake 的单位四元数采样，均匀采样 SO(3)
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()
    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    # map to (w, x, y, z)
    w, x, y, z = q4, q1, q2, q3
    # 旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def rotation_matrix_z(theta_rad):
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def process_directory_with_rotations(directory_path, rotations_per_sample, mode, out_suffix, point_size):
    for root, dirs, files in os.walk(directory_path):
        dirs.sort()
        files.sort()
        for file in files:
            if not file.endswith('.tiff'):
                continue
            tiff_path = os.path.join(root, file)
            file_id = os.path.splitext(file)[0]
            rgb_path = tiff_path.replace('xyz', 'rgb').replace('.tiff', '.png')
            gt_path = tiff_path.replace('xyz', 'gt').replace('.tiff', '.png')

            for rot_idx in range(rotations_per_sample):
                if mode == 'z':
                    # 在指定角度范围内采样（0-360 deg）
                    theta = np.random.uniform(0, 2*math.pi)
                    R = rotation_matrix_z(theta)
                    rot_tag = f"z_{int(math.degrees(theta))}"
                else:
                    R = random_rotation_matrix_full()
                    rot_tag = f"full_{rot_idx}"

                img_save_path = os.path.join(root.replace('xyz', f'2d_rendering_{out_suffix}'), rot_tag, file_id)
                gt_save_path = os.path.join(root.replace('xyz', f'2d_gt_{out_suffix}'), rot_tag, file_id)
                cor_save_path = os.path.join(root.replace('xyz', f'2d_3d_cor_{out_suffix}'), rot_tag, file_id)

                os.makedirs(img_save_path, exist_ok=True)
                os.makedirs(gt_save_path, exist_ok=True)
                os.makedirs(cor_save_path, exist_ok=True)

                print(f"Processing {tiff_path} rot={rot_tag} -> {img_save_path}")
                try:
                    get_mv_images(tiff_path, rgb_path, gt_path, point_size, img_save_path, gt_save_path, cor_save_path, file_id, rotation_matrix=R)
                except Exception as e:
                    print(f"Failed for {tiff_path} rot={rot_tag}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--rotations_per_sample', type=int, default=1)
    parser.add_argument('--mode', choices=['z', 'full'], default='full', help='z = single-axis z rotation; full = full SO(3)')
    parser.add_argument('--out_suffix', type=str, default='rot_exp')
    parser.add_argument('--point_size', type=int, default=7)
    args = parser.parse_args()

    # 只处理 test 子目录，保持与主脚本一致
    for cls in os.listdir(args.data_path):
        cls_path = os.path.join(args.data_path, cls)
        test_dir = os.path.join(cls_path, 'test')
        if os.path.isdir(test_dir):
            print(f"Start class {cls}")
            process_directory_with_rotations(test_dir, args.rotations_per_sample, args.mode, args.out_suffix, args.point_size)

    print('Done')
