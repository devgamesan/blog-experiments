# Torchvision Mask R-CNN インスタンスセグメンテーション

Torchvision の高精度な `maskrcnn_resnet50_fpn_v2` を利用するインスタンスセグメンテーション Notebook の専用環境です。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## セットアップ

このディレクトリで実行します。`uv sync` はこのディレクトリ専用の `.venv` を作成し、`uv.lock` に固定された依存関係をインストールします。

```bash
cd object-detection/maskrcnn-segmentation
uv sync
uv run python -m ipykernel install --user \
  --name maskrcnn-segmentation \
  --display-name "Mask R-CNN Segmentation (Python 3.12)"
uv run jupyter lab
```

JupyterLab で `maskrcnn_segmentation.ipynb` を開き、`Mask R-CNN Segmentation (Python 3.12)` カーネルを選択してください。

## 入出力とモデル

- 入力画像: `../assets/test1.png`
- モデル: Torchvision `maskrcnn_resnet50_fpn_v2`（COCO事前学習済み重み）
- モデル重み: Notebook の初回実行時にTorchvisionの標準キャッシュへ自動ダウンロードされます（約177 MB）。重みファイルはこのリポジトリで配布しません。
- 出力: Notebook上でマスク、バウンディングボックス、クラスラベルを可視化します。

CUDAが利用可能な環境ではGPUを自動利用し、それ以外ではCPUで推論します。GPUを利用する場合は、利用環境に対応するPyTorchをインストールしてから `uv lock` と `uv sync` を更新してください。

## ライセンスと事前学習済み重み

Torchvision本体は[BSD-3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE)です。COCO事前学習済み重みには個別の利用条件が適用される場合があるため、利用前に[公式Torchvisionリポジトリ](https://github.com/pytorch/vision)および関連する公式情報を確認してください。
