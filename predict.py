import warnings
import os
from pathlib import Path
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
import json
import numpy as np

# 忽略警告
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# 初始化模型
model = RFDETRBase(pretrain_weights=r"E:\project\麦穗\outputs\checkpoint_best_total.pth",
                   num_classes=1, 
                   resolution=560, 
                   device='cuda', 
                   use_ema=True, 
                   gradient_checkpointing=True,
                   class_names=["plant"])

# 读取json文件
json_path = r'E:\project\麦穗\coco_dataset\test\_annotations.coco.json'
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# 创建保存结果的文件夹
output_dir = Path(r'E:\project\麦穗\predict_results')
output_dir.mkdir(exist_ok=True)

# 使用自定义类别标签
CUSTOM_CLASSES = ["plant"]  # 与训练配置一致

# 遍历json文件中的所有图像
for img_info in coco_data['images']:
    # 读取图像
    image_path = Path(r'E:\project\麦穗\coco_dataset\test') / img_info['file_name']
    image = Image.open(image_path)
    
    # 预测
    detections = model.predict(image, threshold=0.5)
    
    # 处理预测结果
    valid_detections = [
        (max(0, min(class_id, len(CUSTOM_CLASSES)-1)), confidence)
        for class_id, confidence 
        in zip(detections.class_id, detections.confidence)
    ]
    
    labels = [
        f"{CUSTOM_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence 
        in valid_detections
    ]
    
    # 获取当前图像的标注数据
    gt_boxes = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_info['id']:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
    gt_boxes = np.array(gt_boxes)
    
    # 可视化结果
    annotated_image = image.copy()
    box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF0000"))  # 红色
    annotated_image = box_annotator.annotate(
        scene=annotated_image,
        detections=detections
    )
    
    # 绘制未识别目标的绿色框
    if len(gt_boxes) > 0:
        gt_detections = sv.Detections(
            xyxy=gt_boxes,
            class_id=np.array([0] * len(gt_boxes)),
            confidence=np.array([1.0] * len(gt_boxes))
        )
        gt_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#00FF00"))  # 绿色
        annotated_image = gt_box_annotator.annotate(
            scene=annotated_image,
            detections=gt_detections
        )
    
    # 保存结果
    output_path = output_dir / img_info['file_name']
    annotated_image.save(output_path)
    print(f"已保存: {output_path}")