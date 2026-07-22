# model.py
import torch
import torch.nn as nn
# ★ _b / _B_ ではなく、すべて _s / _S_ (Small) に変更します
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

def get_rugby_mvit_model(num_classes=2):
    """MViTv2(Small)モデルを構築し、2クラス用にカスタマイズして返す"""
    weights = MViT_V2_S_Weights.DEFAULT
    model = mvit_v2_s(weights=weights)
    
    in_features = model.head[1].in_features
    model.head[1] = nn.Linear(in_features, num_classes)
    
    transforms = weights.transforms()
    return model, transforms