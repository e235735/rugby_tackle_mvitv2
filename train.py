import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RugbyVideoDataset
from model import get_rugby_mvit_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def set_seed(seed=42):
    """乱数シードの完全固定関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """DataLoader用ワーカーシード固定処理"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model():
    seed_value = 42
    set_seed(seed_value)
    
    g = torch.Generator()
    g.manual_seed(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    model, transforms = get_rugby_mvit_model(num_classes=2)
    model = model.to(device)
    
    train_dataset = RugbyVideoDataset(root_dir="dataset/train_500Tackles2", transform=transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    num_epochs = 10
    print("学習を開始します...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n学習が完了しました。モデルの重みを保存します...")
    torch.save(model.state_dict(), "rugby_mvit_model.pth")
    print("保存が完了しました！ファイル名: rugby_mvit_model.pth")

if __name__ == "__main__":
    train_model()