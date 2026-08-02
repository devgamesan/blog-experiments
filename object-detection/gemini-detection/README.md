# Gemini セグメンテーション物体検出・カウント

Google Gemini APIで画像内の主要な物体を検出し、インスタンスごとのセグメンテーション輪郭を可視化してラベル別に数えるNotebookの専用環境です。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Google Gemini APIキーと、画像理解・セグメンテーションに対応したGeminiモデル

## セットアップ

このディレクトリで実行します。

```bash
cd object-detection/gemini-detection
uv sync
uv run python -m ipykernel install --user \
  --name gemini-detection \
  --display-name "Gemini Detection (Python 3.12)"
uv run jupyter lab
```

JupyterLabで`gemini_detection.ipynb`を開き、`Gemini Detection (Python 3.12)`カーネルを選択してください。

## API設定

APIキーやモデル名をNotebookやリポジトリへ書き込まず、起動前に環境変数またはこのディレクトリの`.env`ファイルへ設定します。Notebookは`.env`を読み込みますが、既に設定済みの環境変数を上書きしません。

```bash
cp .env.example .env
```

```bash
export GEMINI_API_KEY='your-api-key'
export GEMINI_MODEL='your-gemini-model'
```

- `GEMINI_API_KEY`: Google Gemini APIの認証キー
- `GEMINI_MODEL`: 画像理解・セグメンテーションを利用するGeminiモデル名

## 入出力と制約

- 入力画像: `../assets/test1.png`
- 出力: セグメンテーション輪郭を重ねた画像、検出一覧、ラベル別件数をNotebook内に表示します。ファイル保存はしません。
- Geminiには、英語ラベル`label`と日本語ラベル`label_ja`、`box_2d`（`[ymin, xmin, ymax, xmax]`）、`mask`（0〜1000正規化の`[[x, y], ...]`輪郭ポリゴン）を含むJSONを要求します。可視化と集計では日本語・英語を併記し、可視化の主結果はポリゴンです。
- Geminiの物体検出・セグメンテーションは生成AIの出力です。専用のインスタンスセグメンテーションモデルが返す較正済みのスコアや厳密なマスクとしては扱わず、重要な用途では目視確認してください。
