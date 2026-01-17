import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir

mvtec_3d_classes = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                    'foam', 'peach', 'potato', 'rope', 'tire']
_C = edict()

# common
_C.seed = 7351

# data
_C.data = edict()
_C.data.dataset_root = '/data2/ZTT/mvtec_3d_mv'#'./data/mvtec3d' 
_C.data.class_name = 'bagel'
_C.data.num_points = 4096  # 每个样本在训练阶段目标点数（整数），用于二分搜索体素尺寸和下采样目标
_C.data.rotation_magnitude = 360.0  # 随机旋转幅度
_C.data.translation_magnitude = 0.5  # 随机平移幅度
_C.data.keep_ratio = 0.7  # 裁剪/采样保留比例
_C.data.voxel_size = 0.03  # 体素尺寸（浮点数），用于点云体素化
_C.data.crop_method = "plane"  # 裁剪方法（字符串），如 "plane" 或 "point"，决定如何从点云中裁切子区域。
_C.data.twice_sample = True
_C.data.twice_transform = False

# dirs
_C.root_dir = osp.dirname(osp.realpath(__file__))
_C.output_dir = osp.join(_C.root_dir, "output_mvtec3d")  # 训练输出根目录（保存结果/模型）
_C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8  # PyTorch DataLoader 的 worker 数量
_C.train.noise_magnitude = 0.02  # 训练时对点云添加的高斯噪声标准差

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 1

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0  # 接受配准时的重叠阈值（float），用于评估时判定配准是否具有足够重叠。
_C.eval.acceptance_radius = 0.1  # 接受配准时的距离阈值（float，坐标单位），配准点被视为 inlier 的距离上限。
_C.eval.inlier_ratio_threshold = 0.03  # 判定成功配准所需的内点比例阈值（float），例如 0.03 表示至少 3% 点为 inlier。
_C.eval.rre_threshold = 1.0  # 相对旋转误差阈值（度数），评估旋转精度的门限。
_C.eval.rte_threshold = 0.1  # 平移误差阈值（坐标单位），评估平移精度的门限。

# ransac 用于推理配准
_C.ransac = edict()
_C.ransac.num_points = 3  # RANSAC 每次采样的点对数
_C.ransac.num_iterations = 50000  # RANSAC 最大迭代次数

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps = 10000
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.1
_C.optim.max_iteration = 100000
_C.optim.snapshot_steps = 10000
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 3
_C.backbone.nsample = 32
_C.backbone.in_channel = 64
_C.backbone.input_dim = 32
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.group_norm = 32
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.num_points_in_patch = 128
_C.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 128
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 512
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3  # 从 patch soft-scores 中每侧取 top-k 候选对应，用于构建二值对应矩阵与后续 RANSAC。
_C.fine_matching.confidence_threshold = 0.03  # 置信度阈值，低于该值的匹配被视为不可靠（对 mutual/topk 过滤生效）。
_C.fine_matching.mutual = True  # 是否要求 mutual consistency
_C.fine_matching.use_dustbin = False  # 是否启用 dustbin（额外的“未匹配”行/列）在最终对应矩阵处理阶段。

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0
_C.loss.weight_in_loss = 1.0

def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()