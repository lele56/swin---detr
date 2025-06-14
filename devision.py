#!/usr/bin/env python
import os
import json
import glob
import shutil
import random
import argparse
from PIL import Image


# 支持多类别的YAML文件
try:
    import yaml
except ImportError:
    yaml = None

def convert_yolo_to_coco(image_paths, labels_dir, class_name="custom", add_dummy=False, categories=None):
    """
    将YOLO格式的标注转换为COCO格式的JSON
    
    YOLO标注文件的每行格式应为:
      <类别索引> <x中心点> <y中心点> <宽度> <高度>
    其中坐标是相对于图像尺寸的归一化值
    
    如果提供categories参数，则认为是多类别数据集
      此时YOLO类别索引会被用作COCO类别ID(加1)
    
    对于单类别数据集(当categories为None时)，转换会为所有标注分配相同的类别ID(1)
    除非启用dummy参数，此时会添加一个背景类(id 0)并将目标类别设为id 1
    """
    if categories is None:
        # 单类别转换
        if add_dummy:
            categories = [
                {"id": 0, "name": "Workers", "supercategory": "none"},
                {"id": 1, "name": class_name, "supercategory": "Workers"}
            ]
        else:
            categories = [{"id": 1, "name": class_name, "supercategory": "none"}]
        use_multiclass = False
    else:
        # 多类别: 使用提供的类别
        use_multiclass = True

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    image_id = 1
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # 获取图像尺寸
        with Image.open(img_path) as img:
            width, height = img.size
        
        coco["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        # YOLO标注文件: 相同文件名但扩展名为.txt
        base, _ = os.path.splitext(filename)
        label_file = os.path.join(labels_dir, base + ".txt")
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # 跳过格式不正确的行
                
                # 多类别模式下使用YOLO类别索引
                if use_multiclass:
                    cls_idx = min(int(parts[0]), 89)  # 限制最大类别ID为89
                    category_id = cls_idx + 1
                else:
                    category_id = 1  # 单类别固定为1
                
                # 解析并转换归一化坐标
                _, x_center, y_center, w_norm, h_norm = map(float, parts)
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                w_abs = w_norm * width
                h_abs = h_norm * height
                x_min = x_center_abs - (w_abs / 2)
                y_min = y_center_abs - (h_abs / 2)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, w_abs, h_abs],
                    "area": w_abs * h_abs,
                    "iscrowd": 0
                }
                coco["annotations"].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    return coco

def create_coco_dataset_from_yolo(yolo_dataset_dir, coco_dataset_dir, class_name="custom",
                                  add_dummy=False, split_ratios=(0.8, 0.1, 0.1), categories=None):
    """
    将YOLO格式的数据集(包含"images"和"labels"子目录)转换为COCO格式
    
    输出目录将包含三个子目录: "train", "valid"和"test"
    每个子目录包含:
      - 对应的图像文件副本
      - "_annotations.coco.json"文件包含COCO格式的标注
    
    如果提供categories参数，则认为是多类别数据集
    """
    images_dir = os.path.join(yolo_dataset_dir, "images")
    labels_dir = os.path.join(yolo_dataset_dir, "labels")
    
    # 收集图像文件路径(支持常见图像扩展名)
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    # 如果不是Windows系统(区分大小写)，添加大写扩展名
    #if sys.platform != "win32":                                                             
    #    image_extensions += tuple(ext.upper() for ext in image_extensions)
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if not image_paths:
        raise ValueError("在指定的images目录中未找到任何图像文件")
    
    # 打乱图像顺序并划分为train/valid/test集
    random.shuffle(image_paths)
    num_images = len(image_paths)
    train_end = int(split_ratios[0] * num_images)
    valid_end = train_end + int(split_ratios[1] * num_images)
    
    splits = {
        "train": image_paths[:train_end],
        "valid": image_paths[train_end:valid_end],
        "test": image_paths[valid_end:]
    }
    
    os.makedirs(coco_dataset_dir, exist_ok=True)
    
    for split_name, paths in splits.items():
        split_dir = os.path.join(coco_dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # 复制图像到子目录
        for img_path in paths:
            shutil.copy(img_path, os.path.join(split_dir, os.path.basename(img_path)))
        
        # 转换该子集的标注
        coco_annotations = convert_yolo_to_coco(paths, labels_dir, class_name=class_name,
                                                  add_dummy=add_dummy, categories=categories)
        json_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(json_path, "w") as f:
            json.dump(coco_annotations, f, indent=4)
        print(f"已创建 {json_path}: {len(coco_annotations['images'])}张图像, {len(coco_annotations['annotations'])}个标注")
    
    return coco_dataset_dir

def main():
    parser = argparse.ArgumentParser(
        description="将YOLO格式的数据集转换为COCO JSON格式。支持单类别和多类别数据集。对于多类别数据集，需提供YOLO YAML文件"
    )
    parser.add_argument("--yolo_dataset_dir", type=str, required=True,
                        help="YOLO数据集目录路径(应包含'images'和'labels'子目录)")
    parser.add_argument("--coco_dataset_dir", type=str, default="converted_coco_dataset",
                        help="COCO格式数据集输出目录")
    parser.add_argument("--class_name", type=str, default="custom",
                        help="目标类别名称(用于单类别数据集)")
    parser.add_argument("--add_dummy", action="store_true",
                        help="为单类别数据集添加虚拟背景类(用于RF-DETR的变通方案)")
    parser.add_argument("--yaml_file", type=str, default=None,
                        help="YOLO YAML文件路径(用于多类别数据集)")
    args = parser.parse_args()

    # 根据是否提供YAML文件确定类别
    categories = None
    if args.yaml_file:
        if yaml is None:
            raise ImportError("解析YAML文件需要PyYAML库。请通过'pip install pyyaml'安装")
        with open(args.yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        names = yaml_data.get("names")
        if not names:
            raise ValueError("YAML文件中不包含'names'键或类别名称")
        # 多类别数据集创建COCO类别(从1开始)
        categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(names)]
        print(f"从YAML文件加载了{len(categories)}个类别")
    
    print("开始从YOLO格式转换为COCO格式...")
    create_coco_dataset_from_yolo(
        yolo_dataset_dir=args.yolo_dataset_dir,
        coco_dataset_dir=args.coco_dataset_dir,
        class_name=args.class_name,
        add_dummy=args.add_dummy,
        categories=categories
    )
    print("转换完成")

if __name__ == "__main__":
    main()