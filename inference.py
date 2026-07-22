# inference.py
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import get_rugby_mvit_model

def predict_video(video_path):
    # 1. デバイスの設定 (GPUが使えればGPU、なければCPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 2. モデル構造と前処理の準備
    model, transforms = get_rugby_mvit_model(num_classes=2)
    
    # 3. 保存した重みファイルを読み込む
    weights_path = "rugby_mvit_model.pth"
    if not os.path.exists(weights_path):
        print(f"エラー: 重みファイル '{weights_path}' が見つかりません。先に train.py を実行してください。")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()  # 推論モードに設定
    print("学習済みモデルの読み込みに成功しました。")

    # 4. 動画の読み込みと16フレームのサンプリング (dataset.pyと同じロジック)
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
        print(f"エラー: 動画ファイル '{video_path}' を正常に読み込めませんでした。")
        return

    # 16フレームを均等にサンプリング
    num_frames = 16
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    sampled_frames = [all_frames[i] for i in indices]
    
    # 5. テンソル変換と前処理
    tensors = []
    for f in sampled_frames:
        rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1) # (C, H, W)
        tensors.append(tensor_frame)
        
    video_tensor = torch.stack(tensors) # (T, C, H, W)
    
    # 公式の前処理を適用
    video_tensor = transforms(video_tensor) # (T, C, H, W)
    
    # モデルが期待する形状 (C, T, H, W) に変換
    if video_tensor.size(0) != 3 and video_tensor.size(1) == 3:
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
    # バッチの次元を追加 (1, C, T, H, W) してデバイスへ送る
    video_tensor = video_tensor.unsqueeze(0).to(device)

    # 6. AIによる推論
    class_names = ["background", "tackle"]
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(video_tensor)
            probabilities = F.softmax(outputs, dim=1)[0] # 確率に変換
            
    # 7. 結果の表示
    prediction_idx = torch.argmax(probabilities).item()
    prediction_class = class_names[prediction_idx]
    confidence = probabilities[prediction_idx].item() * 100

    print("\n" + "="*30)
    print(f"🎬 対象動画: {os.path.basename(video_path)}")
    print(f"🤖 判定結果: {prediction_class.upper()}")
    print(f"🎯 確信度  : {confidence:.2f} %")
    print("="*30)
    
    print(f"\n詳細確率 -> background: {probabilities[0].item()*100:.1f}%, tackle: {probabilities[1].item()*100:.1f}%")

if __name__ == "__main__":
    # テストしたい動画のパスをここに指定してください
    test_video = "dataset/val/background/other_0038.mp4" # ← 実際の動画ファイル名に書き換えてください
    
    if os.path.exists(test_video):
        predict_video(test_video)
    else:
        print(f"指定されたテスト動画が見つかりません: {test_video}")
        print("dataset/val/ の中にある実際の動画パスに書き換えて実行してください。")