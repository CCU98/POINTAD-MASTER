import argparse
import os
import sys
import glob
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

# current_dir = os.path.dirname(os.path.abspath(__file__))
# reg2inv_root = os.path.join(current_dir, 'Reg2Inv-main')
# sys.path.insert(0, reg2inv_root)

from config_mvtec3d import make_cfg
from model import create_model
from geotransformer.utils.data import precompute_data_stack_mode_two
from geotransformer.utils.torch import to_cuda
from dataset_mvtec3d import train_valid_data_loader 


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.remove_non_finite_points()
    return np.asarray(pcd.points, dtype=np.float32), pcd

def save_pcd(points, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True) # ASCII方便查看，Binary更小

class Reg2InvInfer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        self.cfg = make_cfg()
        
        # 尝试获取 neighbor limits 和 voxel size (模拟 Loader 行为)
        # 这里为了简单，我们使用 MVTec 默认推荐值，避免加载整个Loader慢
        self.neighbor_limits = [38, 36, 36] 
        self.voxel_size = 0.03 # 必须与训练时一致
        
        print(f"[Init] Creating model (Voxel: {self.voxel_size})...")
        self.model = create_model(self.cfg, self.voxel_size, self.neighbor_limits)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[Init] Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict['model'], strict=True)
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def infer(self, ref_points, src_points):
        ref_points = torch.from_numpy(ref_points)
        src_points = torch.from_numpy(src_points)
        ref_len = torch.LongTensor([ref_points.shape[0]])
        src_len = torch.LongTensor([src_points.shape[0]])
        
        # 手动计算 search_radius
        search_radius = self.voxel_size * 2.5
        # 获取 num_stages，默认 4
        num_stages = getattr(self.cfg.backbone, 'num_stages', 4)

        input_dict = precompute_data_stack_mode_two(
            ref_points, 
            src_points, 
            ref_len, 
            src_len,
            num_stages, 
            self.voxel_size,
            search_radius,  # 使用计算值
            self.neighbor_limits
        )
        
        input_dict = to_cuda(input_dict)
        with torch.no_grad():
            output_dict = self.model(input_dict)
            est_transform = output_dict['estimated_transform'] 
            return est_transform.squeeze(0).cpu().numpy()

def process_mvtec_dataset(dataset_root, inferencer):
    """
    遍历 MVTec-3D 目录结构并进行配准
    结构: class/test/defect_type/pcd/*.pcd
    """
    # 1. 获取所有类别 (folder)
    classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()
    
    print(f"Found classes: {classes}")

    for cls_name in classes:
        print(f"\n========== Processing Class: {cls_name} ==========")
        cls_root = os.path.join(dataset_root, cls_name)
        
        # 1. 寻找 Template (通常在 train/good/000.pcd 或 *template.pcd)
        train_dir = os.path.join(cls_root, 'train', 'good','pcd') # 或者是 train/
        template_candidates = glob.glob(os.path.join(train_dir, '*.pcd'))
        # 优先找名字里带 template 的，没有就找 000.pcd，再没有就随便拿第一个
        template_path = None
        for p in template_candidates:
            if 'template' in p: template_path = p; break
        if not template_path and template_candidates:
            template_candidates.sort()
            template_path = template_candidates[0] # 通常是 000.pcd
            
        if not template_path:
            print(f"[Skip] No template found for {cls_name} in {train_dir}")
            continue
            
        print(f"Using Template: {os.path.basename(template_path)}")
        ref_points, _ = load_pcd(template_path)
        
        # 2. 遍历 test 文件夹下的所有子文件夹 (good, cut, crack, hole ...)
        test_root = os.path.join(cls_root, 'test')
        if not os.path.exists(test_root):
            continue
            
        defect_types = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
        
        for defect in defect_types:
            # 定位源 pcd 目录: class/test/defect/pcd/
            src_pcd_dir = os.path.join(test_root, defect, 'pcd')
            if not os.path.exists(src_pcd_dir):
                continue
                
            # 定位输出目录: class/test/defect/register_pcd/
            
            output_dir = os.path.join(test_root, defect, 'register_pcd')
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            pcd_files = glob.glob(os.path.join(src_pcd_dir, '*.pcd'))
            print(f"  -> Processing {defect} ({len(pcd_files)} files)... Output: {output_dir}")
            
            for pcd_path in tqdm(pcd_files, leave=False):
                filename = os.path.basename(pcd_path)
                save_path = os.path.join(output_dir, filename)
                
                # 如果已经存在且不想覆盖，可以 continue
                # if os.path.exists(save_path): continue

                try:
                    src_points, _ = load_pcd(pcd_path)
                    
                    # 推理
                    transform = inferencer.infer(ref_points, src_points)
                    
                    # 应用变换
                    R = transform[:3, :3]
                    t = transform[:3, 3]
                    aligned_points = np.dot(src_points, R.T) + t
                    
                    # 保存
                    save_pcd(aligned_points, save_path)
                    
                except Exception as e:
                    print(f"[Error] Failed on {filename}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='MVTec-3D 根目录, e.g., ./data/mvtec_3d_mv')
    parser.add_argument('--checkpoint', type=str, required=True, help='Reg2Inv 通用模型权重路径')
    args = parser.parse_args()

    # 初始化模型
    inferencer = Reg2InvInfer(args.checkpoint)
    
    # 开始批量处理
    process_mvtec_dataset(args.dataset_root, inferencer)
    
    print("\nAll Done! Generated 'register_pcd' folders inside each test category.")

if __name__ == "__main__":
    main()

""""
python batch_register.py \
  --dataset_root /data2/ZTT/mvtec_3d_mv \
  --checkpoint output_mvtec3d/snapshots/iter-80000.pth.tar
"""