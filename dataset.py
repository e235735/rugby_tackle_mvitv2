# dataset.py
import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# ★ NumPyが正常になれば、この公式標準のインポートが絶対に通ります
from torchvision.models.video import MViT_V2_S_Weights

class RugbyVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.class_to_idx = {"background": 0, "tackle": 1}
        self.video_paths = []
        self.labels = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for ext in ('*.mp4', '*.avi', '*.mov', '*.MP4', '*.mpeg'):
                for video_path in glob.glob(os.path.join(class_folder, ext)):
                    self.video_paths.append(video_path)
                    self.labels.append(class_idx)
                    
        print(f"[{root_dir}] から合計 {len(self.video_paths)} 本の動画を読み込みました。")

    def __len__(self):
        return len(self.video_paths)

    def _load_and_sample_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        
        total_frames = len(all_frames)
        if total_frames == 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
            
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        sampled_frames = [all_frames[i] for i in indices]
        return sampled_frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._load_and_sample_video(video_path)
        
        tensors = []
        for f in frames:
            rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1) # (C, H, W)
            tensors.append(tensor_frame)
        
        # 1. まずは公式前処理が期待する (T, C, H, W) の形状でスタックする
        video_tensor = torch.stack(tensors) 
        
        if self.transform:
            # 2. ここで公式の前処理を実行（形状は T, C, H, W のまま受け取らせる）
            video_tensor = self.transform(video_tensor)
            
            # 3. 前処理が終わった後、もし形状が (C, T, H, W) になっていなければここで並び替える
            if video_tensor.size(0) != 3 and video_tensor.size(1) == 3:
                video_tensor = video_tensor.permute(1, 0, 2, 3)
        else:
            # トランスフォームがない場合のフォールバック
            video_tensor = video_tensor.float() / 255.0
            video_tensor = video_tensor.permute(1, 0, 2, 3) 
            
        return video_tensor, label

if __name__ == "__main__":
    print("dataset.py の動作テストを開始します...")
    mvit_transforms = MViT_V2_S_Weights.DEFAULT.transforms()
    target_dir = "rugby_dataset/train"
    if not os.path.exists(target_dir):
        print(f"'{target_dir}' が見つかりません。")
    else:
        dataset = RugbyVideoDataset(root_dir=target_dir, transform=mvit_transforms)