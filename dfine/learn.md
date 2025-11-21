# DINOv3 编码器替换学习笔记

> 目标：在 D-FINE 中用 DINOv3 作为编码器主干，保持总参数≈25M，并确保可在 8 卡服务器上稳定训练。

---

## 1. 预备知识
- **现有结构**：`DFINE -> HGNetv2 backbone -> HybridEncoder`。HybridEncoder 期待 3 个尺度 (stride 8/16/32)、256 通道的特征。
- **DINOv3 特点**：自监督 ViT，默认输出单尺度 token，stride=patch_size (如 ViT-S/14 ≈ 14)；需额外模块生成多尺度特征。
- **参考论文**：
  - DINOv3 技术报告 (arXiv:2508.10104)
  - Real-Time Object Detection Meets DINOv3 (arXiv:2509.20787)
  - SegDINO (arXiv:2509.00833)

---

## 2. 资源准备
1. **权重**：从 HuggingFace 拉取 `facebook/dinov3-small` (约 23M 参数)；如需离线部署可提前下载到 `weight/dinov3/`。
2. **依赖**：确认 `pip install -r requirements.txt`，若使用 HuggingFace Hub 需 `huggingface_hub`；可选 `timm>=1.0`。
3. **代码入口**：新建 `src/nn/backbone/dinov3_backbone.py`；在 `src/nn/backbone/__init__.py` 中注册。

---

## 3. 模型设计 (25M 参数预算)
| 模块 | 方案 | 预估参数 |
| --- | --- | --- |
| Backbone | DINOv3-Small (ViT-S/14) | ~23M |
| Spatial Tuning Adapter (STA) | 1×1 Conv + 上/下采样模块 | ~1M |
| HybridEncoder + DFINE Transformer | 原配置 (略调) | ~1M |
| **合计** |  | **≈25M** |

### STA 设计要点
- 将 ViT token map reshape 为 `B×C×H/14×W/14`。
- `P8`: 最近邻上采样 2× + 1×1 Conv (384→256)。
- `P16`: 1×1 Conv 直接映射。
- `P32`: 3×3 stride-2 Conv 或 AvgPool + Conv。
- 输出数组 `[P8, P16, P32]`，传至 HybridEncoder。

---

## 4. 实施步骤
1. **封装 ViT Backbone**
   - 使用 `timm.create_model("dinov3_small", pretrained=True, features_only=False)` 或官方脚本加载。
   - 通过自定义 `forward_features` 获取最后一层 token (包含 CLS token)；丢弃 CLS，只保留 patch tokens。
   - reshape：`tokens = tokens[:, 1:, :].transpose(1, 2)` → `B×C×H×W`。

2. **构建 STA/FPN** (`src/nn/backbone/modules/dinov3_sta.py`)
   - 三个分支分别负责 P8/P16/P32。
   - 引入可选 LayerNorm + 1×1 Conv 以稳定训练。
   - 输出通道统一为 256，便于接入现有 HybridEncoder。

3. **更新 Config**
   - 复制 `configs/dfine/dfine_hgnetv2_l_coco.yml` 为 `configs/dfine/dfine_dinov3s_coco.yml`。
   - 修改：
     ```yaml
     DFINE:
       backbone: DINOv3Backbone
     DINOv3Backbone:
       model_name: 'dinov3_small'
       pretrained: True
       out_channels: [256, 256, 256]
       feat_strides: [8, 16, 32]
     HybridEncoder:
       in_channels: [256, 256, 256]
       feat_strides: [8, 16, 32]
     ```
   - 调整 optimizer：骨干学习率设 1e-5，其余保持 2.5e-4；backbone norm 权重 decay=0。

4. **训练策略**
   - Phase A (0-10 epoch)：冻结 backbone (`requires_grad=False`)，只训练 STA + DFINE；加速收敛。
   - Phase B：逐层解冻最后 4 个 ViT block，学习率衰减 0.1。
   - 建议命令：
     ```bash
     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
     torchrun --master_port=29500 --nproc_per_node=8 \
       train.py -c configs/dfine/dfine_dinov3s_coco.yml \
       --use-amp --seed=0
     ```
   - 每卡 batch=2，梯度累计 1；若显存不足，使用 `--opt-level O2` 或梯度累积 2。

5. **验证与推理**
   - 保存 best.pth 后执行：
     ```bash
     torchrun --nproc_per_node=8 train.py \
       -c configs/dfine/dfine_dinov3s_coco.yml \
       --test-only -r output/dfine_dinov3s_coco/best.pth
     ```
   - 检查 `loss_vfl`、`loss_bbox` 曲线是否平稳；比较 AP 与 HGNetv2 基线。

---

## 5. 常见问题
| 问题 | 解决思路 |
| --- | --- |
| 梯度爆炸或 loss nan | 使用 AMP、降低 backbone LR、在 STA 前添加 LayerNorm。 |
| 显存不足 | 冻结更多 ViT 层、减小 batch、启用梯度检查点。 |
| 多尺度特征失真 | STA 中加入可学习位置编码或使用 `F.interpolate(mode="bilinear")`。 |
| 收敛慢 | 适当延长冻结阶段，或在 HybridEncoder 中提升 `num_encoder_layers`。 |

---

## 6. 后续优化方向
1. 引入 Gram Anchoring（DINOv3 提出的正则）保持密集特征稳定。
2. 尝试 LoRA/Adapter Tuning，进一步控制参数。
3. 将 `num_queries` 从 300 降至 200，降低显存和计算。
4. 评估更大 patch (ViT-M/16) 与 Objects365 预训练，观察泛化提升。

---

## 7. Checklist
- [ ] `DINOv3Backbone` 类实现并注册。
- [ ] STA 模块输出 3 个尺度，通道=256。
- [ ] Config/optimizer/EMA 更新就绪。
- [ ] 8 卡训练命令可运行，日志正常。
- [ ] 推理/评估脚本验证精度。

完成以上步骤，即可用 DINOv3 成功替换 DFINE 的编码器，并在 8 卡服务器上满足 25M 参数预算的训练需求。祝实验顺利！
