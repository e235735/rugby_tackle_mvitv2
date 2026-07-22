# realtime_tackle_detector.py
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import get_rugby_mvit_model

def process_realtime_stream(input_video_path, output_video_path="output_realtime.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 1. モデルと重みのロード
    model, transforms = get_rugby_mvit_model(num_classes=2)
    weights_path = "rugby_mvit_model.pth"
    if not os.path.exists(weights_path):
        print(f"エラー: '{weights_path}' が見つかりません。先に train.py を実行してください。")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("モデルの読み込みに成功しました。リアルタイム描画を開始します...")

    # 2. 動画の読み込みと出力設定
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 解析結果を書き出す設定 (MP4形式)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # パラメータ設定
    window_duration = 2.0  # AIに見せる長さ (2.0秒)
    window_frames = int(fps * window_duration)
    infer_interval = 5     # 5フレームごとに1回 AI推論を実行 (描画の高速化・滑らか化)

    ring_buffer = []
    frame_idx = 0
    raw_tackle_prob = 0.0      # AIが算出した最新の確率
    display_tackle_prob = 0.0  # メーターアニメーション用のスムーズな確率

    print(f"動画解析中... (出力先: {output_video_path})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        ring_buffer.append(frame.copy())
        frame_idx += 1

        # 1. 2秒分のフレームが蓄積されたらAI推論（指定インターバルごと）
        if len(ring_buffer) >= window_frames:
            if frame_idx % infer_interval == 0:
                # 直近2秒分から16フレームを均等サンプリング
                indices = np.linspace(0, len(ring_buffer) - 1, 16).astype(int)
                sampled_frames = [ring_buffer[i] for i in indices]

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
                        raw_tackle_prob = probabilities[1].item()

            # バッファの長さを2秒分に保つ
            ring_buffer.pop(0)

        # 2. 画面表示用確率のスムージング補間（数値が滑らかに上下移動する演出）
        display_tackle_prob = display_tackle_prob * 0.7 + raw_tackle_prob * 0.3

        # 3. 画面（HUD）のオーバーレイ描画
        annotated_frame = frame.copy()

        # --- [UI 1: 半透明の背景パネル] ---
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        # --- [UI 2: 状態に応じたカラー判定] ---
        if display_tackle_prob < 0.4:
            bar_color = (0, 255, 0)     # 緑 (通常)
            status_text = "NORMAL"
        elif display_tackle_prob < 0.7:
            bar_color = (0, 255, 255)   # 黄 (コンタクト/予兆)
            status_text = "CAUTION"
        else:
            bar_color = (0, 0, 255)     # 赤 (タックル検出)
            status_text = "TACKLE!"

        # --- [UI 3: テキスト & 確率プログレスバー] ---
        cv2.putText(annotated_frame, f"RUGBY AI - MONITOR", (35, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"{display_tackle_prob * 100:5.1f}% [{status_text}]", (35, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # カラーバー（外枠 ＆ 中身）
        bar_max_width = 380
        current_bar_width = int(bar_max_width * display_tackle_prob)
        cv2.rectangle(annotated_frame, (35, 95), (35 + current_bar_width, 115), bar_color, -1)
        cv2.rectangle(annotated_frame, (35, 95), (35 + bar_max_width, 115), (255, 255, 255), 2)

        # --- [UI 4: 70%以上の場合の赤枠警告フラッシュ ＆ 下部バナー] ---
        if display_tackle_prob >= 0.7:
            # 外枠を赤い太線で囲む
            cv2.rectangle(annotated_frame, (0, 0), (width, height), (0, 0, 255), 12)
            
            # 画面下部に巨大警告バナーを表示
            banner_y = height - 60
            cv2.rectangle(annotated_frame, (width//2 - 220, banner_y - 35), (width//2 + 220, banner_y + 15), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "TACKLE DETECTED!", (width//2 - 190, banner_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # 動画ファイルへ書き出し
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\n🎉 リアルタイム風の映像生成が完了しました！")
    print(f"生成された動画ファイル: {output_video_path}")

if __name__ == "__main__":
    # 解析・描画したい動画のパスを指定してください
    target_video = "dataset/101.mp4" # 👈 手元の動画パスに変更
    
    if os.path.exists(target_video):
        process_realtime_stream(target_video, "output_realtime.mp4")
    else:
        print(f"動画が見つかりません: {target_video}")