import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RugbyVideoDataset
from model import get_rugby_mvit_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルと前処理の取得
    model, transforms = get_rugby_mvit_model(num_classes=2)
    model = model.to(device)
    
    # データの準備
    train_dataset = RugbyVideoDataset(root_dir="dataset/train", transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    num_epochs = 10
    print("学習を開始します...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * videos.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
        torch.cuda.empty_cache()

    # ★★★【追加】学習終了後にモデルの重みを保存する ★★★
    print("\n学習が完了しました。モデルの重みを保存します...")
    torch.save(model.state_dict(), "rugby_mvit_model.pth")
    print("保存が完了しました！ファイル名: rugby_mvit_model.pth")

if __name__ == "__main__":
    train_model()