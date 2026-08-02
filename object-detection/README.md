# 物体検出サンプル

画像内の対象を検出・可視化する複数の手法を比較するためのJupyter Notebookサンプルです。用途に応じて、専用の物体検出モデル、テキストプロンプトで検出するモデル、画像を理解できるVLMを試せます。

## サンプル一覧

| ディレクトリ | 手法 | 概要 | Notebook |
| --- | --- | --- | --- |
| [yolo-detection](yolo-detection/README.md) | YOLOv4-p6 + OpenCV DNN | 物体のクラスとバウンディングボックスを検出します。 | `yolo_object_detection.ipynb` |
| [maskrcnn-segmentation](maskrcnn-segmentation/README.md) | Torchvision Mask R-CNN | 物体ごとの領域マスクを含むインスタンスセグメンテーションを行います。 | `maskrcnn_segmentation.ipynb` |
| [rtdetr-detection](rtdetr-detection/README.md) | RT-DETR | TransformerベースでCOCO 80分類の物体を検出し、クラス別に数えます。 | `rtdetr_object_detection.ipynb` |
| [sahi-rtdetr-counting](sahi-rtdetr-counting/README.md) | SAHI + RT-DETR | 重複タイル推論で小物を検出し、統合後の候補をクラス別に数えます。 | `sahi_rtdetr_counting.ipynb` |
| [grounding-dino](grounding-dino/README.md) | Grounding DINO | テキストプロンプトで指定した対象を検出します。 | `grounding_dino_detection.ipynb` |
| [vlm-detection](vlm-detection/README.md) | OpenAI互換VLM | 画像対応のChat Completions APIで主要な物体を検出・可視化します。 | `vlm_object_detection.ipynb` |
| [vlm-grounding-dino](vlm-grounding-dino/README.md) | OpenAI互換VLM + Grounding DINO | VLMで抽出した物体名をGrounding DINOの入力にして検出します。 | `vlm_grounding_dino_detection.ipynb` |
| [vlm-grounding-dino-sam2-counting](vlm-grounding-dino-sam2-counting/README.md) | OpenAI互換VLM + Grounding DINO + SAM2 | カテゴリを自動発見し、マスクと候補検証を使って物体をカテゴリ別に数えます。 | `vlm_grounding_dino_sam2_counting.ipynb` |

## 共通事項

- 入力画像には[`assets/test1.png`](assets/test1.png)を使用します。
- 各サンプルは独立したPython 3.12・[uv](https://docs.astral.sh/uv/)環境です。実行前に対象ディレクトリへ移動し、個別READMEに従ってセットアップしてください。
- VLMサンプルのみ、画像入力に対応したOpenAI互換APIと認証情報の設定が必要です。環境変数や`.env`の設定方法は[個別README](vlm-detection/README.md)を参照してください。
- モデル重み、モデルキャッシュ、仮想環境、推論結果などの生成物はGit管理しません。

## 使い分け

- 定義済みの物体クラスを精度優先で検出したい場合は、YOLOv4-p6 + OpenCV DNNの物体検出を使用します。
- 物体の輪郭・領域も扱いたい場合は、Torchvision Mask R-CNNのインスタンスセグメンテーションを使用します。
- COCO 80分類を Transformer ベースの検出器で検出・集計したい場合は、RT-DETRを使用します。
- 高解像度画像の小物・遠景物体を数えたい場合は、SAHI + RT-DETRのタイル推論を使用します。タイル数に応じて処理時間が増え、密着物体の完全な分離は保証されません。
- 任意の語句で検出対象を指定したい場合は、Grounding DINOを使用します。
- API経由で画像の内容を柔軟に解析したい場合は、OpenAI互換VLMを使用します。VLMが返す座標やconfidenceは専用検出器の較正済み出力ではありません。
- 画像ごとに検出候補を自動生成しつつ、専用検出器の位置とスコアを使いたい場合は、OpenAI互換VLMとGrounding DINOの統合サンプルを使用します。
- カテゴリを指定せず、物体の輪郭、重複、部品・背景・描画物体の可能性まで確認してカテゴリ別に数えたい場合は、VLM + Grounding DINO + SAM2のカウントサンプルを使用します。
