# RT-DETR 物体検出・カウント

Apache-2.0 の `PekingU/rtdetr_r101vd` を使い、COCO 80分類の物体を検出して、クラス別に数える Jupyter Notebook です。RT-DETR は Transformer ベースのエンドツーエンド検出器で、既存の YOLO・Mask R-CNN と異なる方式を比較できます。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## セットアップと Jupyter ランタイムへの追加

このディレクトリで実行します。`uv sync` は専用の `.venv` を作成し、次のコマンドはその仮想環境を Jupyter のカーネルとして登録します。

```bash
cd object-detection/rtdetr-detection
uv sync
uv run python -m ipykernel install --user \
  --name rtdetr-detection \
  --display-name "RT-DETR Detection (Python 3.12)"
uv run jupyter lab
```

JupyterLab で `rtdetr_object_detection.ipynb` を開き、`RT-DETR Detection (Python 3.12)` カーネルを選択してください。

## 入出力とモデル

- 入力画像: `../assets/test1.png`
- モデル: [`PekingU/rtdetr_r101vd`](https://huggingface.co/PekingU/rtdetr_r101vd)（COCO 80分類、Apache-2.0、精度重視）
- 出力: 検出ボックス付き画像、クラス別件数、検出詳細表

モデル重みは初回実行時に Hugging Face の標準キャッシュへ取得され、リポジトリには保存しません。CUDA 対応 PyTorch が利用できる環境では GPU を自動使用します。

## ライセンス

このリポジトリは MIT License です。RT-DETR のコードと上記公式モデル重み、Transformers は Apache-2.0 です。再配布時は各依存物のライセンス表示要件と、学習データセット COCO の利用条件を確認してください。
