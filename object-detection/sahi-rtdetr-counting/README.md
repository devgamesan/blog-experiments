# SAHI + RT-DETR 小物検出・カウント

RT-DETR で通常推論と SAHI のタイル推論を行い、統合後の検出結果をクラス別に数える Jupyter Notebook です。大きな画像内の小物・遠景物体を、画像を分割して検出することで拾いやすくします。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## セットアップと Jupyter ランタイムへの追加

```bash
cd object-detection/sahi-rtdetr-counting
uv sync
uv run python -m ipykernel install --user \
  --name sahi-rtdetr-counting \
  --display-name "SAHI + RT-DETR Counting (Python 3.12)"
uv run jupyter lab
```

JupyterLab で `sahi_rtdetr_counting.ipynb` を開き、`SAHI + RT-DETR Counting (Python 3.12)` カーネルを選択してください。

## 入出力と使い方

- 入力画像: `../assets/test1.png`
- 検出器: [`PekingU/rtdetr_r101vd`](https://huggingface.co/PekingU/rtdetr_r101vd)（COCO 80分類、精度重視）
- 出力: 通常推論とタイル推論の件数比較、タイル推論で統合済みの検出ボックス、クラス別件数、検出詳細表

Notebook の先頭セルにある `SLICE_HEIGHT`、`SLICE_WIDTH`、重なり率、信頼度閾値を対象画像に合わせて調整してください。タイル数が増えるほど小物を拾える可能性は高まりますが、処理時間も増えます。SAHI はタイル境界で生じる重複候補を統合しますが、密着した物体を必ず個別に分離するものではありません。

モデル重みと SAHI の生成物はリポジトリへ保存しません。CUDA 対応 PyTorch が利用できる環境では GPU を自動使用します。

## ライセンス

このリポジトリは MIT License です。SAHI は MIT License、RT-DETR の公式コード・モデル重みと Transformers は Apache-2.0 です。再配布時は各依存物のライセンス表示要件と COCO の利用条件を確認してください。
