# YOLO 物体検出

OpenCV DNN でYOLOv4系列の最高精度モデルである YOLOv4-p6 を利用する物体検出 Notebook の専用環境です。Darknet 本体のインストール・ビルドや追加のPythonパッケージは必要ありません。

## モデル選定

最新のUltralytics YOLO11は[AGPL-3.0ライセンス](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)で提供されています。このサンプルでは、そのライセンス条件を避けて利用したいケースを想定し、Darknet形式のYOLOv4-p6をOpenCV DNNで読み込みます。Darknetリポジトリのライセンスと、学習済み重み・学習データセットの利用条件は、[ライセンスと利用条件](#ライセンスと利用条件)を確認してください。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## セットアップ

このディレクトリで実行します。`uv sync` はこのディレクトリ専用の `.venv` を作成し、`uv.lock` に固定された依存関係をインストールします。

```bash
cd object-detection/yolo-detection
uv sync
uv run python -m ipykernel install --user \
  --name yolo-detection \
  --display-name "YOLO Detection (Python 3.12)"
uv run jupyter lab
```

JupyterLab で `yolo_object_detection.ipynb` を開き、`YOLO Detection (Python 3.12)` カーネルを選択してください。

## 入出力とモデル

- 入力画像: `../assets/test1.png`
- モデル: COCO 80クラスで学習済みの YOLOv4-p6（入力サイズ 1280×1280）
- モデルファイル: Notebook の初回実行時に `models/yolov4-p6/` へ自動ダウンロードされます。
  - 設定: [`yolov4-p6.cfg`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg)
  - 重み（約487MB）: [`yolov4-p6.weights`](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.weights)
  - クラス名: [`coco.names`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)
- 出力: Notebook上の検出結果画像、クラス別集計、検出詳細表

`models/yolov4-p6/` の設定・重み・クラス名、出力画像、`.venv` は Git 管理しません。ダウンロードに失敗した場合はネットワーク接続と保存先の書き込み権限を確認し、Notebookのモデル取得セルを再実行してください。精度を優先するモデルのため、CPU推論には時間がかかります。

## ライセンスと利用条件

このNotebookはDarknetを実行せず、OpenCV DNNでDarknet形式のモデルファイルを読み込みます。Darknetリポジトリの[LICENSE](https://raw.githubusercontent.com/AlexeyAB/darknet/master/LICENSE)は、Darknetをパブリックドメインとして扱っています。

学習済み重みはリポジトリへ同梱せず、上記の公式配布元から実行環境へ直接取得します。重みおよびその学習に使われたデータセットの利用可否・再配布条件は、利用者が配布元と関連するデータセットの条件を確認してください。
