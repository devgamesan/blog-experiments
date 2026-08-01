# VLM + Grounding DINO 物体検出

OpenAI互換VLMで画像内の主要な物体カテゴリを抽出し、その名称をGrounding DINOのテキストプロンプトとして渡すNotebook環境です。VLMは検出候補の発見、Grounding DINOはバウンディングボックスと検出スコアの算出を担当します。

## 機能

- **カテゴリ発見**: VLMから `label`（見た目の名前）、`canonical_label`（集計用の統一名）、`aliases`（別名）を含むカテゴリ情報を取得
- **別名検索**: `label`、`aliases`、`canonical_label` をすべて検索語として Grounding DINO に渡し、見逃しを削減
- **統一集計**: 検出結果を `canonical_label` で統一して集計

## 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- 画像入力に対応したOpenAI互換VLM API

## セットアップとJupyterカーネル登録

このディレクトリで実行します。`uv sync` は、このサンプル専用の`.venv`を作成して依存関係をインストールします。

```bash
cd object-detection/vlm-grounding-dino
uv sync
uv run python -m ipykernel install --user \
  --name vlm-grounding-dino \
  --display-name "VLM + Grounding DINO (Python 3.12)"
uv run jupyter lab
```

JupyterLabで`vlm_grounding_dino_detection.ipynb`を開き、`VLM + Grounding DINO (Python 3.12)`カーネルを選択してください。このカーネルは、このディレクトリで作成したvenvを実行環境として使用します。

## API設定

APIキーをNotebookやリポジトリへ書き込まず、起動前に環境変数またはこのディレクトリの`.env`ファイルへ設定します。Notebookは`.env`を読み込みますが、既に設定済みの環境変数を上書きしません。

`.env.example`をコピーして`.env`を作成する場合:

```bash
cp .env.example .env
```

```bash
export VLM_API_KEY='your-api-key'
export VLM_MODEL='your-vision-model'
```

- `VLM_API_KEY`: OpenAI互換APIの認証キー
- `VLM_MODEL`: 画像入力をサポートするモデル名
- `VLM_BASE_URL`: OpenAI以外のOpenAI互換APIを使う場合だけ指定する、`/v1`を含むAPIベースURL。未設定時はOpenAI SDKの既定接続先を使用します。

## 入出力と調整

- 入力画像: `../assets/test1.png`
- Grounding DINOモデル: `IDEA-Research/grounding-dino-base`。初回実行時にHugging Faceからダウンロードされます（約936MB）。
- VLMはカテゴリ情報（`label`、`canonical_label`、`aliases`）を返し、Notebookはこれらをすべて検索語として `object one. object two. alias one. canonical one.` 形式のプロンプトへ変換します。
- 検出結果は `canonical_label` で統一されて集計されます。例：`coffee mug` と `cup with handle` は両方とも `mug` として集計されます。
- 既定の閾値は `BOX_THRESHOLD=0.35`、`TEXT_THRESHOLD=0.25` です。見逃しが多い場合は下げ、誤検出が多い場合は上げてください。
- VLMの応答はモデル依存です。VLMが返すカテゴリは候補であり、最終的な位置・スコアはGrounding DINOの出力を確認してください。

GPU対応PyTorchが利用できる環境では自動的にGPUで推論します。モデルキャッシュ、`.venv`、`.env`はGit管理しません。
