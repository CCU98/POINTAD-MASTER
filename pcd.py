import open3d as o3d
import numpy as np

pcd_path = "001.pcd"  # 替换为你的路径
pcd = o3d.io.read_point_cloud(pcd_path)

# 1. 简单的布尔值检查
if pcd.has_colors():
    print("✅ 这个 PCD 文件包含颜色信息。")
    
    # 2. 进一步检查颜色数据是否为空白（全黑或全白）
    colors = np.asarray(pcd.colors)
    print(f"颜色数组形状: {colors.shape}")  # 应该是 (N, 3)
    print(f"前5个点的颜色值:\n{colors[:5]}")
else:
    print("❌ 这个 PCD 文件不包含颜色信息（只有坐标）。")