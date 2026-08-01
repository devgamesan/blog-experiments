# Grounding DINO 物体検出

テキストプロンプトで指定した対象を検出する Grounding DINO の Notebook 環境です。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Grounding DINO は Hugging Face Transformers 版を使用します。独自の C++/CUDA 拡張のビルドは不要で、CUDA 対応PyTorchが利用できる環境では自動的にGPUで推論します。互換性を確認済みの `transformers<5` に固定しています。

## セットアップ

このディレクトリで実行します。`uv sync` はこのディレクトリ専用の `.venv` を作成し、`uv.lock` に固定された依存関係をインストールします。

```bash
cd object-detection/grounding-dino
uv sync
uv run python -m ipykernel install --user \
  --name grounding-dino \
  --display-name "Grounding DINO (Python 3.12)"
uv run jupyter lab
```

JupyterLab で `grounding_dino_detection.ipynb` を開き、`Grounding DINO (Python 3.12)` カーネルを選択してください。

## 入出力とモデル

- 入力画像: `../assets/test1.png`
- モデル: `IDEA-Research/grounding-dino-base`。Notebook の初回実行時に Hugging Face から自動ダウンロードされます（約936MB）。
- テキストプロンプト: 小文字で、各クラス名を `.` で区切って指定します。

モデルキャッシュと `.venv` は Git 管理しません。GPUを利用する場合は、利用環境に対応するCUDA対応PyTorchをインストールしてから `uv lock` と `uv sync` を更新してください。
