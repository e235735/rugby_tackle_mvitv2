# 🏉 Rugby Tackle Detector (MViTv2)

---

## 📁 ディレクトリ・ファイル構成

```text
rugby_tackle_mvitv2/
├── dataset/                   # 学習・評価用およびテスト用の動画データ
│   ├── train/                 # モデル学習用データ
│   │   ├── background/        # タックル以外の動画（other_0001.mp4 〜）
│   │   └── tackle/            # タックルシーン動画（tackle_0001.mp4 〜）
│   └── val/                   # モデル評価・検証用データ
│       ├── background/        # 検証用：背景動画
│       └── tackle/            # 検証用：タックル動画
│
├── dataset.py                 # データセット読み込み・前処理クラス（PyTorch Dataset）
├── model.py                   # MViTv2 (Small) モデルの定義およびクラス数の調整
├── train.py                   # モデルの学習ルーチン（重みの保存を含む）
├── inference.py               # 単一の短尺動画に対する推論・判定スクリプト
├── detect_long_video.py       # 長時間動画をスライディングウィンドウでスキャン解析
├── realtime_tackle_detector.py # 映像に確率メーターや警告UIを合成して動画出力
├── manual_clipper.py # 動画を切り抜いてクリップにする
│
└── requirements.txt           # 依存ライブラリ一覧