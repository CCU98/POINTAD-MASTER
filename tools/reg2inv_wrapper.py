"""简易 Reg2Inv 推理 wrapper，供渲染脚本按需调用（在线或批量）。
示例：
from tools.reg2inv_wrapper import Reg2InvWrapper
WRAPPER = Reg2InvWrapper(reg_root='Reg2Inv-main', checkpoint='output_mvtec3d/snapshots/latest.pth')
T = WRAPPER.estimate_transform(ref_pts_np, src_pts_np)
"""
import sys
import os
import numpy as np


class Reg2InvWrapper:
    def __init__(self, reg_root='Reg2Inv-main', checkpoint=None, device='cuda'):
        self.reg_root = reg_root
        self.checkpoint = checkpoint
        self.device = device
        self._impl = None
        if reg_root is not None:
            sys.path.insert(0, os.path.abspath(reg_root))
            try:
                from tools.reg2inv_preprocess import Reg2InvInferWrapper
                self._impl = Reg2InvInferWrapper(reg_root, checkpoint, device)
            except Exception:
                self._impl = None

    def estimate_transform(self, ref_pts_np, src_pts_np):
        if self._impl is None:
            return None
        return self._impl.estimate_transform(ref_pts_np, src_pts_np)
