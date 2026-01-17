import os
import argparse
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path

def generate_grey_pcd(tiff_path):
    """
    生成与 PointAD 论文及你日志一致的 PCD 文件：
    1. 仅保留几何信息 (XYZ)。
    2. 剔除背景 (0,0,0)。
    3. 强制颜色为统一灰色 [0.70196078, 0.70196078, 0.70196078]。
    """
    tiff_path = Path(tiff_path)
    
    # ================= 1. 确定保存路径 =================
    # 保存到同级的 pcd 文件夹中 (例如 .../train/good/pcd/000.pcd)
    current_dir = tiff_path.parent
    save_dir = str(current_dir).replace('xyz', 'pcd')
    os.makedirs(save_dir, exist_ok=True)
    
    save_name = tiff_path.name.replace('.tiff', '.pcd')
    save_path = os.path.join(save_dir, save_name)

    # ================= 2. 读取几何数据 (TIFF) =================
    # 读取 XYZ 数据 (HxWx3)
    organized_pc = tiff.imread(str(tiff_path))
    
    # ================= 3. 数据清洗 (Flatten & Filter) =================
    # 展平为 (N, 3)
    points = organized_pc.reshape(-1, 3)
    
    # 创建掩码：只保留有坐标的点 (去除预处理时填充的 0 背景)
    # 只要 x, y, z 中有一个不为 0，就认为是有效点
    valid_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_mask]

    # ================= 4. 创建点云并上色 =================
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    # 【关键步骤】强制统一上色，匹配你的日志数值
    # 0.70196078 对应的是 RGB(179, 179, 179)
    grey_value = 179 / 255.0  # 约等于 0.7019607843
    pcd.paint_uniform_color([grey_value, grey_value, grey_value])

    # (可选) 计算法向量，这对 3D 渲染和 PointAD 的渲染步骤可能有帮助
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # ================= 5. 保存 =================
    # write_ascii=True 方便你用 head 命令查看文件头确认数值
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Paper-Consistent Grey PCDs')
    parser.add_argument('dataset_path', type=str, help='Root path of dataset (e.g., ./mvtec_3d/bagel)')
    args = parser.parse_args()

    root_path = args.dataset_path
    
    # 递归查找所有 tiff 文件
    tiff_files = list(Path(root_path).rglob('*.tiff'))
    total_files = len(tiff_files)
    
    print(f"Found {total_files} tiff files.")
    print(f"Generating uniform grey PCDs (Color: {179/255.0:.8f})...")
    
    count = 0
    for path in tiff_files:
        generate_grey_pcd(path)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total_files}...")
            
    print("Done! Check your 'pcd' folders.")