import cv2

video_path = "dataset/train/tackle/tackle_0032.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)                      # 1秒あたりのフレーム数
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)     # 総フレーム数
cap.release()

if fps > 0:
    duration = total_frames / fps
    print(f"動画の長さ: {duration:.2f} 秒")