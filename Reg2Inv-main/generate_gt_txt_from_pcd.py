import os
import glob
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm

def generate_txt_only_in_subfolders(dataset_root, category):
    # 基础路径: .../carrot/test
    base_test_path = os.path.join(dataset_root, category, 'test')
    
    print(f"正在处理类别: {category}")
    print(f"扫描目录: {base_test_path}")
    print("模式: 仅生成 .txt 文件，不修改 .pcd")

    # 遍历所有子文件夹 (good, cut, collision, etc.)
    sub_folders = [f.path for f in os.scandir(base_test_path) if f.is_dir()]

    for folder_path in sub_folders:
        defect_type = os.path.basename(folder_path) # e.g. 'cut'
        
        # 如果是 good 类别，不需要生成 txt (因为 dataset 逻辑是通过文件名判断 good 自动设为 label=0)
        if defect_type == 'good':
            continue

        # 定义输入路径 (原始数据)
        xyz_dir = os.path.join(folder_path, 'xyz') # 存放 tiff
        gt_dir = os.path.join(folder_path, 'gt')   # 存放 png
        
        # 定义输出路径: .../test/cut/gt/
        # 注意：这里我们把生成的 txt 放在了该类别下的 gt 子目录里
        output_txt_dir = os.path.join(folder_path, 'gt') 
        
        if not os.path.exists(xyz_dir):
            continue
            
        # 确保输出目录存在 (如果原始 gt 文件夹里全是 png，txt 也会存进去；或者它会创建新文件夹)
        os.makedirs(output_txt_dir, exist_ok=True)

        tiff_files = sorted(glob.glob(os.path.join(xyz_dir, '*.tiff')))
        
        for tiff_path in tqdm(tiff_files, desc=f"Generating TXT for {defect_type}"):
            # 文件名 ID (e.g. '000')
            file_id = os.path.splitext(os.path.basename(tiff_path))[0]
            
            # 1. 读取 TIFF (为了获取几何坐标 x,y,z)
            # 必须读 TIFF 才能保证生成的 txt 里的坐标和你的 pcd 是对应的
            xyz_map = tiff.imread(tiff_path)
            
            # 2. 读取 PNG Mask
            png_path = os.path.join(gt_dir, f"{file_id}.png")
            if os.path.exists(png_path):
                mask = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                # 尺寸对齐
                if mask.shape != xyz_map.shape[:2]:
                    mask = cv2.resize(mask, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                # 如果没有 png，默认全 0
                mask = np.zeros(xyz_map.shape[:2], dtype=np.uint8)

            # 3. 展平并过滤 (核心：只保留非背景点)
            points = xyz_map.reshape(-1, 3)
            labels = mask.reshape(-1)
            
            # 过滤条件：Z != 0 (假设您的 PCD 也是这样生成的)
            valid_idx = points[:, 2] != 0 
            
            valid_points = points[valid_idx]
            valid_labels = (labels[valid_idx] > 0).astype(int)
            
            # 4. 保存 TXT
            # 路径: .../test/cut/gt/000.txt
            txt_save_path = os.path.join(output_txt_dir, f"{file_id}.txt")
            
            # 拼接 x y z label
            data_to_save = np.column_stack((valid_points, valid_labels))
            
            # 保存
            np.savetxt(txt_save_path, data_to_save, fmt='%.6f %.6f %.6f %d', delimiter=' ')

    print("\n所有 TXT 生成完毕！")

if __name__ == "__main__":
    # 请修改为您的实际路径
    DATASET_ROOT = "/data2/ZTT/mvtec_3d_mv"
    CATEGORY = "carrot"
    
    generate_txt_only_in_subfolders(DATASET_ROOT, CATEGORY)