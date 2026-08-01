# OpenAI互換VLM 物体検出

OpenAI互換のChat Completions APIで画像を解析し、画像内の主要な物体をバウンディングボックスとして可視化するNotebookの専用環境です。

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- 画像入力に対応したOpenAI互換VLM API

## セットアップ

このディレクトリで実行します。

```bash
cd object-detection/vlm-detection
uv sync
uv run python -m ipykernel install --user \
  --name vlm-detection \
  --display-name "VLM Detection (Python 3.12)"
uv run jupyter lab
```

JupyterLabで`vlm_object_detection.ipynb`を開き、`VLM Detection (Python 3.12)`カーネルを選択してください。

## API設定

APIキーをNotebookやリポジトリへ書き込まず、起動前に環境変数またはこのディレクトリの`.env`ファイルへ設定します。Notebookは`.env`を読み込みますが、既に設定済みの環境変数を上書きしません。

```bash
export VLM_API_KEY='your-api-key'
export VLM_BASE_URL='https://your-api-provider.example/v1'
export VLM_MODEL='your-vision-model'
```

`.env`を使う場合は、`.env.example`をコピーして`vlm-detection/.env`を作成してください。このファイルはGit管理しません。

```bash
cp .env.example .env
```

- `VLM_API_KEY`: OpenAI互換APIの認証キー
- `VLM_BASE_URL`: `/v1` を含むAPIベースURL
- `VLM_MODEL`: 画像入力をサポートするモデル名

## 入出力と制約

- 入力画像: `../assets/test1.png`
- 出力: 注釈付き画像、検出一覧、ラベル別件数をNotebook内に表示します。ファイル保存はしません。
- VLMが返す矩形座標とconfidenceは、専用物体検出器の出力ではありません。confidenceはモデルの自己申告による参考値であり、YOLOの信頼度のような較正済みスコアとして扱わないでください。
