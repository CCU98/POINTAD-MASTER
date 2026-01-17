**Reg2Inv 集成到渲染前（PointAD）**

- 概要：先运行离线预处理把点云配准并保存，再用 `multi_view` 的渲染脚本渲染配准后的点云进行 A/B 比较。

- 步骤（示例）：

1) 离线配准（示例命令）：

```bash
python tools/reg2inv_preprocess.py \
  --input_dir path/to/original/xyz \
  --output_dir path/to/xyz_registered \
  --reg2inv_root Reg2Inv-main \
  --checkpoint output_mvtec3d/snapshots/snapshot.pth.tar \
  --device cuda \
  --ref_template path/to/canonical_template.ply
```

2) 使用配准后的点云渲染：
- 将渲染脚本中的输入目录替换为 `path/to/xyz_registered`（或修改 `multiview_mvtec.py` 的 `input_dir` 参数），然后运行渲染脚本（项目 README 中提供的渲染命令）。

3) 对比实验：渲染并运行 PointAD 测试脚本（`test.py` 或项目提供的测试流程），分别统计原始与配准后结果进行 A/B 比较。

- 注意事项：
  - 如果 Reg2Inv 模型无法直接加载或接口不匹配，脚本会回退到 PCA 对齐（可按需换成 PCA+ICP）。
  - 建议先在少量样本上验证，观察是否保留局部异常信号。
