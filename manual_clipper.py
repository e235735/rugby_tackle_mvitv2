# manual_clipper.py
import os
import cv2
import numpy as np

INSTRUCTIONS = """
==================================================
              改良版 Rugby Clipper
==================================================
【動画再生・移動コントロール】
  - [Space]キー    : 再生 / 一時停止
  - [D] キー       : 1フレーム 進む (微調整用)
  - [A] キー       : 1フレーム 戻る (微調整用)
  - [L] キー       : 30フレーム進む (★約1秒スキップ)
  - [K] キー       : 30フレーム戻る (★約1秒バック)
  ※ 画面下のシークバーをマウスでドラッグして一気ジャンプも可能！

【アノテーション（一時停止中に実行）】
  - [T] キー       : タックル (tackle) として切り出し
  - [O] キー       : その他 (other) として切り出し
  
【切り出し手順】
  1. 目的のシーンの「中心」で [Space] を押して一時停止。
  2. [T] または [O] を押すと範囲選択モードになります。
  3. マウスで選手をドラッグして囲み、[Enter]キー で確定。
     (間違えた場合は [C]キー でキャンセル)

【終了】
  - [Q] キー       : アプリを終了
==================================================
"""

# トラックバー（シークバー）用のコールバック関数
def on_trackbar(val):
    global frame_idx, g_changed_by_trackbar
    if g_changed_by_trackbar:
        frame_idx = val

def run_manual_clipper(video_path, output_dir, target_fps=30, clip_duration=1.5):
    global frame_idx, g_changed_by_trackbar
    print(INSTRUCTIONS)
    
    # フォルダの作成
    tackle_dir = os.path.join(output_dir, "tackle")
    other_dir = os.path.join(output_dir, "other")
    os.makedirs(tackle_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video not found {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    clip_frames_count = int(target_fps * clip_duration) # 30 * 1.5 = 45 frames
    half_window = clip_frames_count // 2

    frame_idx = 0
    playing = False  # 最初は一時停止状態でスタートして探せるように変更
    g_changed_by_trackbar = True

    tackle_count = len(os.listdir(tackle_dir))
    other_count = len(os.listdir(other_dir))

    window_name = "Rugby Clipper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # シークバーの作成
    cv2.createTrackbar("Position", window_name, 0, total_frames - 1, on_trackbar)

    while True:
        # 再生中、または一時停止中の現在のフレーム読み込み
        if playing:
            ret, frame = cap.read()
            if not ret:
                playing = False
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            # 再生に合わせてシークバーの位置を更新（コールバックのループを防ぐ）
            g_changed_by_trackbar = False
            cv2.setTrackbarPos("Position", window_name, frame_idx)
            g_changed_by_trackbar = True
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

        # 画面に情報をオーバーレイ描画
        display_frame = frame.copy()
        status = "PLAYING" if playing else "PAUSED"
        current_sec = frame_idx / video_fps
        total_sec = total_frames / video_fps
        
        # テキストの装飾を少し見やすく修正
        cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames} ({current_sec:.1f}s/{total_sec:.1f}s) | {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Saved -> Tackles: {tackle_count} | Others: {other_count}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)

        # キー入力受付
        key = cv2.waitKey(20 if playing else 0) & 0xFF

        if key == ord('q'): # 終了
            break
        elif key == ord(' '): # 再生・一時停止
            playing = not playing
        elif key == ord('d'): # 1フレーム進む
            playing = False
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord('a'): # 1フレーム戻る
            playing = False
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('l'): # ★30フレーム（約1秒）進む
            playing = False
            frame_idx = min(total_frames - 1, frame_idx + 30)
        elif key == ord('k'): # ★30フレーム（約1秒）戻る
            playing = False
            frame_idx = max(0, frame_idx - 30)
            
        elif key in [ord('t'), ord('o')]: # クロップ処理
            is_tackle = (key == ord('t'))
            label_name = "tackle" if is_tackle else "other"
            playing = False
            
            print(f"\n[{label_name.upper()} モード] マウスで選手を囲み、Enterで確定してください。")
            bbox = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = bbox
            
            if w > 0 and h > 0:
                start_f = max(0, frame_idx - half_window)
                end_f = min(total_frames - 1, frame_idx + half_window)
                
                frames_to_save = []
                for f in range(start_f, end_f + 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    success, f_img = cap.read()
                    if success:
                        crop = f_img[y:y+h, x:x+w]
                        if crop.size > 0:
                            frames_to_save.append(cv2.resize(crop, (224, 224)))
                        else:
                            frames_to_save.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                while len(frames_to_save) < clip_frames_count:
                    frames_to_save.append(frames_to_save[-1] if len(frames_to_save) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
                
                if is_tackle:
                    save_path = os.path.join(tackle_dir, f"tackle_{tackle_count:04d}.mp4")
                    tackle_count += 1
                else:
                    save_path = os.path.join(other_dir, f"other_{other_count:04d}.mp4")
                    other_count += 1
                    
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, video_fps, (224, 224))
                for f_save in frames_to_save:
                    out.write(f_save)
                out.release()
                print(f"-> 1.5秒動画を保存しました: {save_path}")
            else:
                print("-> キャンセルされました。")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 前回修復した動画のパスを指定
    run_manual_clipper(
        video_path="dataset/videos/30.mov",
        output_dir="dataset/train",
        clip_duration=1.5
    )