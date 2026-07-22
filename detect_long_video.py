# detect_long_video.py
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import get_rugby_mvit_model

def analyze_long_video(video_path, threshold=0.50): # 💡確認のため閾値を50%に下げてみます
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 1. モデルと前処理の準備
    model, transforms = get_rugby_mvit_model(num_classes=2)
    weights_path = "rugby_mvit_model.pth"
    if not os.path.exists(weights_path):
        print(f"エラー: '{weights_path}' がありません。先に train.py を実行してください。")
        return
    
    # 警告を非表示にする設定でロード
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 2. 動画の情報の取得
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print(f"エラー: 動画 '{video_path}' を正常に読み込めませんでした。パスやファイルを確認してください。")
        return
        
    duration = total_frames / fps
    print(f"🎬 動画情報 -> 総フレーム数: {total_frames} | FPS: {fps:.1f} | 長さ: {duration:.1f}秒")

    # スキャンの設定
    window_duration = 2.0  
    stride_duration = 0.5  
    window_frames = int(fps * window_duration)
    stride_frames = int(fps * stride_duration)
    
    # 💡 動画が短すぎる場合のセーフティ
    if total_frames < window_frames:
        print(f"⚠️ 警告: 動画の長さ({duration:.1f}秒)が、スキャン窓({window_duration}秒)より短いためスキャンできません。")
        print("この動画の解析には inference.py を使用するか、もっと長い動画を指定してください。")
        return

    frame_buffer = []
    frame_count = 0
    inference_count = 0
    detections = []
    max_tackle_prob = 0.0
    
    print(f"分析設定   -> {window_duration}秒の窓を、{stride_duration}秒ずつずらして解析（閾値: {threshold*100:.0f}%）")
    print("="*60)
    print(f"{'時間区間 (秒)':<18} | {'タックルである確率':<12} | {'判定結果':<10}")
    print("-"*60)

    # 3. スライディングウィンドウによる解析
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_buffer.append(frame)
        frame_count += 1
        
        if len(frame_buffer) == window_frames:
            inference_count += 1
            end_sec = frame_count / fps
            start_sec = end_sec - window_duration
            
            # 16フレームサンプリング
            indices = np.linspace(0, window_frames - 1, 16).astype(int)
            sampled_frames = [frame_buffer[i] for i in indices]
            
            tensors = []
            for f in sampled_frames:
                rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1)
                tensors.append(tensor_frame)
            
            video_tensor = torch.stack(tensors)
            video_tensor = transforms(video_tensor)
            
            if video_tensor.size(0) != 3 and video_tensor.size(1) == 3:
                video_tensor = video_tensor.permute(1, 0, 2, 3)
                
            video_tensor = video_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(video_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
            
            tackle_prob = probabilities[1].item()
            if tackle_prob > max_tackle_prob:
                max_tackle_prob = tackle_prob
            
            # 全区間の確率をリアルタイムに表示（裏で仕事している証拠が見えます）
            status = "🚨 DETECTED!" if tackle_prob >= threshold else "  (Background)"
            print(f"{start_sec:5.1f}s 〜 {end_sec:5.1f}s  | {tackle_prob*100:6.1f} %        | {status}")
            
            if tackle_prob >= threshold:
                detections.append((start_sec, end_sec, tackle_prob))
            
            frame_buffer = frame_buffer[stride_frames:]

    cap.release()
    print("="*60)
    print(f"📈 解析のまとめ")
    print(f"・総判定回数     : {inference_count} 回")
    print(f"・検出された数   : {len(detections)} 箇所")
    print(f"・今回記録した最高タックル確率: {max_tackle_prob*100:.1f} %")
    print("="*60)

if __name__ == "__main__":
    # テストしたい動画のパス
    long_video_path = "dataset/100.mp4" 
    
    if os.path.exists(long_video_path):
        analyze_long_video(long_video_path, threshold=0.60) # 60%以上を検出対象に
    else:
        print(f"指定された動画が見つかりません: {long_video_path}")