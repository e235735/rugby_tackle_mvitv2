import os
import cv2
import numpy as np

INSTRUCTIONS = """
==================================================
              改良・安定化版 Rugby Clipper
==================================================
【動画再生・移動コントロール】
  - [Space]キー    : 再生 / 一時停止
  - [D] キー       : 1フレーム 進む
  - [A] キー       : 1フレーム 戻る
  - [L] キー       : 30フレーム進む (約1秒)
  - [K] キー       : 30フレーム戻る (約1秒)
  ※ シークバーでの一気ジャンプも爆速・安定化！

【アノテーション（一時停止中に実行）】
  - [T] キー       : タックル (tackle) として切り出し
  - [O] キー       : その他 (other) として切り出し
  
【終了】
  - [Q] キー       : アプリを終了
==================================================
"""

def on_trackbar(val):
    global frame_idx, g_changed_by_trackbar
    if g_changed_by_trackbar:
        frame_idx = val

def run_manual_clipper(video_path, output_dir, target_fps=30, clip_duration=1.0):
    global frame_idx, g_changed_by_trackbar
    print(INSTRUCTIONS)
    
    tackle_dir = os.path.join(output_dir, "tackle")
    other_dir = os.path.join(output_dir, "other")
    os.makedirs(tackle_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)

    # --- 1. 動画フレームの一括全読み込み (Ubuntu環境での安定化策) ---
    print(f"動画を読み込んでいます: {video_path} ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video not found {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = target_fps

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        print("Error: 動画フレームを読み込めませんでした。")
        return
    
    print(f"読み込み完了！ 総フレーム数: {total_frames} ({total_frames / video_fps:.1f}秒)")

    # 渡された target_fps と clip_duration に基いて切り出しフレーム数を計算
    clip_frames_count = int(target_fps * clip_duration)
    half_window = clip_frames_count // 2

    frame_idx = 0
    playing = False
    g_changed_by_trackbar = True

    tackle_count = len(os.listdir(tackle_dir))
    other_count = len(os.listdir(other_dir))

    window_name = "Rugby Clipper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Position", window_name, 0, total_frames - 1, on_trackbar)

    while True:
        # メモリからフレームを取得するだけなので動作が破綻しない
        frame = frames[frame_idx].copy()

        # 再生中の処理
        if playing:
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
                playing = False

            g_changed_by_trackbar = False
            cv2.setTrackbarPos("Position", window_name, frame_idx)
            g_changed_by_trackbar = True

        # 情報表示
        status = "PLAYING" if playing else "PAUSED"
        current_sec = frame_idx / video_fps
        total_sec = total_frames / video_fps
        
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} ({current_sec:.1f}s/{total_sec:.1f}s) | {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved -> Tackles: {tackle_count} | Others: {other_count}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(window_name, frame)

        # キー入力受領
        key = cv2.waitKey(20 if playing else 30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d'):
            playing = False
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord('a'):
            playing = False
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('l'):
            playing = False
            frame_idx = min(total_frames - 1, frame_idx + 30)
        elif key == ord('k'):
            playing = False
            frame_idx = max(0, frame_idx - 30)
            
        elif key in [ord('t'), ord('o')]:
            is_tackle = (key == ord('t'))
            label_name = "tackle" if is_tackle else "other"
            playing = False
            
            print(f"\n[{label_name.upper()} モード] マウスで選手を囲み、Enterで確定してください。")
            
            # 元画像でROIを選択
            raw_frame = frames[frame_idx]
            bbox = cv2.selectROI(window_name, raw_frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = bbox
            
            if w > 0 and h > 0:
                start_f = max(0, frame_idx - half_window)
                end_f = min(total_frames - 1, frame_idx + half_window)
                
                frames_to_save = []
                # メモリ上のリストから一括抽出（高速・不具合ゼロ）
                for f in range(start_f, end_f + 1):
                    f_img = frames[f]
                    crop = f_img[y:y+h, x:x+w]
                    if crop.size > 0:
                        frames_to_save.append(cv2.resize(crop, (224, 224)))
                    else:
                        frames_to_save.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                # パディング処理
                while len(frames_to_save) < clip_frames_count:
                    frames_to_save.append(frames_to_save[-1] if len(frames_to_save) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
                
                if is_tackle:
                    save_path = os.path.join(tackle_dir, f"tackle_500_2_{tackle_count:04d}.mp4")
                    tackle_count += 1
                else:
                    # 元コードにあったアンダーバーのタイポも修正 (other_500_2_XXXX.mp4)
                    save_path = os.path.join(other_dir, f"other_500_2_{other_count:04d}.mp4")
                    other_count += 1
                    
                # Linuxで安定しやすい Codec（mp4v または mp4vで開けない場合は mp4v -> mp4v 代替）
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, video_fps, (224, 224))
                for f_save in frames_to_save:
                    out.write(f_save)
                out.release()
                print(f"-> 保存完了: {save_path}")
            else:
                print("-> キャンセルされました。")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_manual_clipper(
        video_path="dataset/500Tackles2/600.mpeg",
        output_dir="dataset/train_500Tackles2",
        clip_duration=1.0
    )