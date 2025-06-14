import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from rfdetr import RFDETRBase
from timm.models import swin_transformer
import torch

# 配置路径
PREDICTED_IMAGES_DIR = Path(r'E:\project\麦穗\coco_dataset\test')
COCO_JSON_PATH = Path(r'E:\project\麦穗\coco_dataset\test\_annotations.coco.json')
OUTPUT_DIR = Path(r'E:\project\麦穗\analysis_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 初始化模型
swin_backbone = swin_transformer.SwinTransformer(
    img_size=560,
    patch_size=4,
    in_chans=3,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.,
    qkv_bias=True,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False
)

model = RFDETRBase(
    backbone=swin_backbone,
    pretrain_weights=r"E:\project\麦穗\outputs_tran\checkpoint_best_total.pth",
    num_classes=1,
    resolution=560,
    device='cuda',
    use_ema=True,
    gradient_checkpointing=True,
    class_names=["plant"]
)

# 加载COCO标注
with open(COCO_JSON_PATH, 'r') as f:
    coco_data = json.load(f)

# 创建输出目录
detected_dir = OUTPUT_DIR / "detected_crops"
detected_dir.mkdir(exist_ok=True)

def process_image(image_path, img_info):
    """处理单张图像并保存预测框截图"""
    # 加载图像
    image = Image.open(image_path)
    
    # 模型预测
    detections = model.predict(image, threshold=0.5)
    pred_boxes = detections.xyxy
    
    # 保存所有预测框截图
    for i, pred_box in enumerate(pred_boxes):
        crop = image.crop(pred_box)
        save_path = detected_dir / f"{img_info['id']}_{i}.jpg"
        crop.save(save_path)
    
    print(f"已保存 {len(pred_boxes)} 个识别结果: {img_info['file_name']}")

# 处理每张图像
for img_info in coco_data['images']:
    image_path = PREDICTED_IMAGES_DIR / img_info['file_name']
    if not image_path.exists():
        continue
    
    process_image(image_path, img_info)

print("所有图像处理完成！")
print(f"识别结果已保存至: {detected_dir}")