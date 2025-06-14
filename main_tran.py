import torch
from rfdetr import RFDETRBase
from timm.models import swin_transformer
from multiprocessing import freeze_support
import os
import json

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

# 创建Swin Transformer骨干网络
swin_backbone = swin_transformer.SwinTransformer(
    img_size=560,
    patch_size=16,
    in_chans=3,
    embed_dim=192,  # 调整为更标准的192维度
    depths=[2, 2, 18, 2],  # 平衡深度(原[2,2,24,2])
    num_heads=[6, 12, 24, 32],  # 保持注意力头配置
    window_size=7,
    mlp_ratio=4.0,  # 调整MLP比例
    qkv_bias=True,
    drop_rate=0.1,  # 降低dropout率
    attn_drop_rate=0.1,  # 降低注意力dropout率
    drop_path_rate=0.3,  # 调整drop path rate
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=True,
    focal_levels=[2, 2, 2, 2],  # 简化focal levels
    focal_windows=[5, 3, 3, 1]  # 调整focal windows
)


# 修改RFDetRBase模型使用Swin Transformer骨干
model = RFDETRBase(
    backbone=swin_backbone,  # 传入自定义骨干网络
    num_classes=1,
    resolution=560,
    device='cuda',
    use_ema=True,
    gradient_checkpointing=True,
    class_names=["plant"]
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
            output_dir=r"E:\project\麦穗\outputs_tran",  # 训练输出目录
            #resume=r"E:\project\麦穗\outputs\checkpoint0034.pth",  # 恢复训练
            epochs=100,
            batch_size=4,
            grad_accum_steps=4,
            lr=8e-5,  # 调整学习率
            lr_encoder=1.5e-5,  # 调整encoder学习率
            weight_decay=1e-5,  # 调整权重衰减
            checkpoint_interval=5,  # 每5epoch保存一次
            tensorboard=True,  # 启用TensorBoard
            wandb=True,  # 启用Weights & Biases
            project="wheat_head_detection",  # W&B项目名
            run="exp1",  # W&B运行名
            early_stopping=True,  # 启用早停
            early_stopping_patience=15,  # 10epoch无改善则停止
            early_stopping_min_delta=0.002,  # mAP改善阈值
            lr_scheduler='cosine',  # 使用余弦退火学习率
            warmup_epochs=5,  # 新增学习率预热
            warmup_factor=0.01,  # 预热初始学习率比例
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