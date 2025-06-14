from rfdetr import RFDETRBase
from multiprocessing import freeze_support
import os
import json

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def validate_coco_annotations(dataset_dir, mode="train"):
    """检查训练集/验证集/测试集的标注文件"""
    # 构建标注文件的路径，假设标注文件名为"_annotations.coco.json"
    ann_file = os.path.join(dataset_dir, mode, "_annotations.coco.json")
    # 打开标注文件并加载JSON数据
    with open(ann_file) as f:
        data = json.load(f)
    
    # 统一检查单类别数据集
    category_ids = [cat['id'] for cat in data['categories']]
    if len(category_ids) != 1 or category_ids[0] != 1:
        raise ValueError(f"{mode}集应只包含一个类别且ID为1")

model = RFDETRBase(
    pretrain_weights="rf-detr-base.pth",
    num_classes=1,
    resolution=560,  # 输入分辨率
    device='cuda',  # 使用GPU加速
    use_ema=True,  # 启用权重EMA平滑
    gradient_checkpointing=True,  # 节省显存
    class_names=["plant"]  # 明确指定类别名称
)
history = []
def callback2(data):
    history.append(data)

model.callbacks["on_fit_epoch_end"].append(callback2)

if __name__ == '__main__':
    freeze_support()
    try:
        # 检查所有数据集的标注
        validate_coco_annotations(r"E:\project\麦穗\coco_dataset", "train")
        validate_coco_annotations(r"E:\project\麦穗\coco_dataset", "valid")  # 验证集
        validate_coco_annotations(r"E:\project\麦穗\coco_dataset", "test")  # 测试集
        
        model.train(
            dataset_dir=r"E:\project\麦穗\coco_dataset",
            output_dir=r"E:\project\麦穗\outputs",  # 训练输出目录
            #resume=r"E:\project\麦穗\outputs\checkpoint0034.pth",  # 恢复训练
            epochs=100,
            batch_size=4,
            grad_accum_steps=4,
            lr=2e-4,  
            lr_encoder=5e-5,  
            weight_decay=5e-5,  
            checkpoint_interval=5,  # 每5epoch保存一次
            tensorboard=True,  # 启用TensorBoard
            wandb=True,  # 启用Weights & Biases
            project="wheat_head_detection",  # W&B项目名
            run="exp1",  # W&B运行名
            early_stopping=True,  # 启用早停
            early_stopping_patience=10,  # 10epoch无改善则停止
            early_stopping_min_delta=0.005,  # mAP改善阈值
            # 数据增强配置
            train_transforms={
                'random_horizontal_flip': True,
                'random_resize': [504, 560, 616],  # 增加更多尺寸选项
                'color_jitter': {'brightness': 0.2, 'contrast': 0.2},  # 增强颜色扰动
                'random_crop': True,
                'random_zoom': (0.8, 1.2),  # 新增缩放增强
                'random_rotate': (-15, 15)  # 新增旋转增强
            }
        )
    except RuntimeError as e:
        print(f"CUDA Error occurred: {e}")
        if "index out of bounds" in str(e):
            print("错误: 请确认所有数据集都已设置为单类别模式")
    except Exception as e:
        print(f"其他错误: {e}")