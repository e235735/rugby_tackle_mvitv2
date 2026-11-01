import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import get_rugby_mvit_model

def process_realtime_stream(input_video_path, output_video_path="output_realtime.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    model, transforms = get_rugby_mvit_model(num_classes=2)
    weights_path = "rugby_mvit_epoch_10.pth"
    if not os.path.exists(weights_path):
        print(f"エラー: '{weights_path}' が見つかりません。先に train.py を実行してください。")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("モデルの読み込みに成功しました。リアルタイム描画を開始します...")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    window_duration = 1.0
    window_frames = int(fps * window_duration)
    infer_interval = 3

    ring_buffer = []
    frame_idx = 0
    raw_tackle_prob = 0.0
    display_tackle_prob = 0.0

    print(f"動画解析中... (出力先: {output_video_path})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        ring_buffer.append(frame.copy())
        frame_idx += 1

        if len(ring_buffer) >= window_frames:
            if frame_idx % infer_interval == 0:
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
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            outputs = model(video_tensor)
                    else:
                        outputs = model(video_tensor)
                        
                    probabilities = F.softmax(outputs, dim=1)[0]
                    raw_tackle_prob = probabilities[1].item()

            ring_buffer.pop(0)

        display_tackle_prob = display_tackle_prob * 0.7 + raw_tackle_prob * 0.3

        annotated_frame = frame.copy()

        # UI 描画処理
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        if display_tackle_prob < 0.4:
            bar_color = (0, 255, 0)
            status_text = "NORMAL"
        elif display_tackle_prob < 0.7:
            bar_color = (0, 255, 255)
            status_text = "CAUTION"
        else:
            bar_color = (0, 0, 255)
            status_text = "TACKLE!"

        cv2.putText(annotated_frame, "RUGBY AI - MONITOR", (35, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"{display_tackle_prob * 100:5.1f}% [{status_text}]", (35, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        bar_max_width = 380
        current_bar_width = int(bar_max_width * display_tackle_prob)
        cv2.rectangle(annotated_frame, (35, 95), (35 + current_bar_width, 115), bar_color, -1)
        cv2.rectangle(annotated_frame, (35, 95), (35 + bar_max_width, 115), (255, 255, 255), 2)

        if display_tackle_prob >= 0.7:
            cv2.rectangle(annotated_frame, (0, 0), (width, height), (0, 0, 255), 12)
            banner_y = height - 60
            cv2.rectangle(annotated_frame, (width//2 - 220, banner_y - 35), (width//2 + 220, banner_y + 15), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "TACKLE DETECTED!", (width//2 - 190, banner_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\n🎉 リアルタイム風の映像生成が完了しました！")
    print(f"生成された動画ファイル: {output_video_path}")

if __name__ == "__main__":
    target_video = "dataset/new/1495.mpeg"
    if os.path.exists(target_video):
        process_realtime_stream(target_video, "dataset/result/test/1495_10e.mp4")
    else:
        print(f"動画が見つかりません: {target_video}")