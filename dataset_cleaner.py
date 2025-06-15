import os
import json
import shutil
from tqdm import tqdm

def clean_dataset(dataset_dir, modes=("train", "valid", "test"), backup=True):
    """
    清理COCO格式数据集中的未标注图像
    
    参数:
        dataset_dir: 数据集根目录
        modes: 要处理的数据集类型 (train/valid/test)
        backup: 是否创建备份
    """
    # 创建备份目录
    if backup:
        backup_dir = os.path.join(dataset_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
    
    for mode in modes:
        print(f"\n正在处理 {mode} 数据集...")
        mode_dir = os.path.join(dataset_dir, mode)
        ann_file = os.path.join(mode_dir, "_annotations.coco.json")
        
        # 读取标注文件
        with open(ann_file) as f:
            data = json.load(f)
        
        # 获取标注文件中记录的图像文件名
        annotated_images = {img['file_name'] for img in data['images']}
        
        # 获取实际文件夹中的图像文件
        actual_images = set()
        for ext in ['.jpg', '.jpeg', '.png']:
            actual_images.update(f for f in os.listdir(mode_dir) if f.lower().endswith(ext))
        
        # 找出未标注的图像
        unannotated_images = actual_images - annotated_images
        if not unannotated_images:
            print(f"{mode} 数据集没有未标注图像")
            continue
            
        print(f"发现 {len(unannotated_images)} 张未标注图像")
        
        # 处理未标注图像
        for img in tqdm(unannotated_images, desc="处理进度"):
            img_path = os.path.join(mode_dir, img)
            
            if backup:
                # 备份图像
                backup_path = os.path.join(backup_dir, f"{mode}_{img}")
                shutil.copy2(img_path, backup_path)
            
            # 删除图像
            os.remove(img_path)
        
        print(f"已移除 {len(unannotated_images)} 张未标注图像")

if __name__ == '__main__':
    # 直接设置参数运行
    dataset_path = r"E:\project\麦穗\coco_dataset"  # 修改为你的数据集路径
    clean_dataset(
        dataset_dir=dataset_path,
        backup=True  # 设为False则不创建备份
    )