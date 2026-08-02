"""Notebook用の物体候補生成・融合・可視化ヘルパー。"""

from __future__ import annotations

import base64
import gc
import json
import math
import mimetypes
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor, pipeline


# Notebook上で見せる文言。内部処理の識別子を、そのまま利用者に見せないための対応表。
STATUS_DISPLAY_NAMES = {
    "pending": "まだ判定していない",
    "confirmed": "数える（確度が高い）",
    "probable": "数える（やや不確か）",
    "uncertain": "数えずに要確認として残す",
    "unknown": "物体らしいが名前を決められない",
    "rejected": "数えない（部品・背景・画像内の絵など）",
}

REVIEW_REASON_DISPLAY_NAMES = {
    "LOW_DETECTOR_SCORE": "場所探しAIの自信が低い",
    "LOW_MASK_QUALITY": "物の輪郭をうまく切り出せていない",
    "UNMATCHED_AUTO_MASK": "名前なしで見つかった領域",
    "POSSIBLE_PART": "別の物の一部かもしれない",
    "LABEL_CONFLICT": "同じ場所に複数の名前の候補がある",
}

JAPANESE_FONT_CANDIDATES = ["Noto Sans CJK JP", "IPAexGothic", "IPA Gothic", "Yu Gothic", "Meiryo"]


def display_status(status: str) -> str:
    return STATUS_DISPLAY_NAMES.get(status, status)


def display_review_reasons(reasons: list[str]) -> str:
    return "、".join(REVIEW_REASON_DISPLAY_NAMES.get(reason, reason) for reason in reasons)


def configure_japanese_font() -> str | None:
    """Matplotlibで日本語を表示できるフォントを選び、選んだ名前を返す。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in JAPANESE_FONT_CANDIDATES if name in available), None)
    if selected:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def japanese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """画像に重ねる文字用の日本語フォント。利用できない環境ではPillowの既定へ戻す。"""
    for family in JAPANESE_FONT_CANDIDATES:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            return ImageFont.truetype(path, size=size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def display_label(item: dict) -> str:
    """英語の集計名に日本語表示名を併記する。日本語名がない場合も表示を壊さない。"""
    label = item.get("canonical_label") or "unknown"
    japanese = str(item.get("display_name_ja") or "").strip()
    return f"{label}（{japanese}）" if japanese else label


def label_key(label: str) -> str:
    """表記揺れ比較用のキー。toy_car / toy-car / toy car を同じ語として扱う。"""
    return " ".join(re.sub(r"[_-]+", " ", str(label).lower()).split())


def canonical_for_label(label: str, canonical_by_label: dict[str, str]) -> str:
    """検索結果やVLM回答を、カテゴリ発見時に決めた統一名へ戻す。"""
    return canonical_by_label.get(label_key(label), " ".join(str(label).lower().split()))


def load_rgb_image(path: Path) -> Image.Image:
    """EXIFの向きを反映したRGB画像を返す。"""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def build_client(api_key: str, base_url: str | None) -> OpenAI:
    options = {"api_key": api_key, "timeout": 60.0, "max_retries": 2}
    if base_url:
        options["base_url"] = base_url.rstrip("/")
    return OpenAI(**options)


def image_data_url(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format, quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def parse_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSONオブジェクトが見つかりません。")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSONオブジェクトではありません。")
    return payload


def call_vlm_json(client: OpenAI, model_name: str, prompt: str, images: list[Image.Image]) -> dict:
    """互換性を優先してChat Completions APIからJSONを取り出す。"""
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_data_url(image)}} for image in images)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
    )
    return parse_json_object(completion.choices[0].message.content or "")


def discover_categories(client: OpenAI, model_name: str, image: Image.Image) -> tuple[list[dict], dict]:
    prompt = """
画像内に物理的に存在し、独立して数えられる物体カテゴリだけを列挙してください。
壁・床・棚・空などの背景、文字・ロゴ・模様、物体の部品、写真・絵・ポスター・画面内に描かれた物体は除外します。
個数は回答しません。英語の短いカテゴリ名を使い、同義語もGrounding DINOで使える英語名にします。
説明やMarkdownを含めず、必ず次のJSONだけを返してください。
{"categories":[{"label":"coffee mug","canonical_label":"mug","display_name_ja":"マグカップ","aliases":["cup with handle"],"confidence":0.9}]}
""".strip()
    payload = call_vlm_json(client, model_name, prompt, [image])
    categories = []
    seen = set()
    for item in payload.get("categories", []):
        if not isinstance(item, dict):
            continue
        label = " ".join(str(item.get("label", "")).lower().split())
        canonical = " ".join(str(item.get("canonical_label") or label).lower().split())
        canonical_key = label_key(canonical)
        if not label or not canonical or canonical_key in seen:
            continue
        aliases = [" ".join(str(value).lower().split()) for value in item.get("aliases", []) if str(value).strip()]
        display_name_ja = " ".join(str(item.get("display_name_ja", "")).split())
        categories.append({"label": label, "canonical_label": canonical, "display_name_ja": display_name_ja, "aliases": list(dict.fromkeys(aliases)), "confidence": float(item.get("confidence", 0.0))})
        seen.add(canonical_key)
    return categories, payload


def labels_for_categories(categories: list[dict]) -> tuple[list[str], dict[str, str]]:
    search_labels = []
    canonical_by_label = {}
    for category in categories:
        for label in [category["label"], *category["aliases"], category["canonical_label"]]:
            key = label_key(label)
            if key not in canonical_by_label:
                search_labels.append(label)
                canonical_by_label[key] = category["canonical_label"]
    return search_labels, canonical_by_label


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, left[2] - left[0]) * max(0, left[3] - left[1]) + max(0, right[2] - right[0]) * max(0, right[3] - right[1]) - intersection
    return intersection / union if union else 0.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    area = inner.sum()
    return float(np.logical_and(inner, outer).sum() / area) if area else 0.0


def nms_detections(detections: list[dict], threshold: float = 0.5) -> list[dict]:
    kept = []
    for candidate in sorted(detections, key=lambda item: item["detector_score"], reverse=True):
        if not any(candidate["canonical_label"] == other["canonical_label"] and bbox_iou(candidate["bbox"], other["bbox"]) >= threshold for other in kept):
            kept.append(candidate)
    return kept


def _grounding_dino_detections(
    image: Image.Image,
    labels: list[str],
    canonical_by_label: dict[str, str],
    device: str,
    box_threshold: float,
    text_threshold: float,
    labels_per_batch: int,
    processor,
    model,
    source: str,
    offset: tuple[int, int] = (0, 0),
) -> tuple[list[dict], list[dict]]:
    """ロード済みのGrounding DINOで1枚（全体画像またはタイル）を検出する。"""
    detections, batches = [], []
    for batch_index in range(0, len(labels), labels_per_batch):
        batch_labels = labels[batch_index : batch_index + labels_per_batch]
        started = perf_counter()
        inputs = processor(images=image, text=". ".join(batch_labels) + ".", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        text_labels = [str(label).lower() for label in result.get("text_labels", [])]
        for box, score, label in zip(boxes, scores, text_labels):
            label = " ".join(label.split())
            x_offset, y_offset = offset
            x1, y1, x2, y2 = box
            detections.append({
                "bbox": [float(x1 + x_offset), float(y1 + y_offset), float(x2 + x_offset), float(y2 + y_offset)],
                "detector_label": label,
                "canonical_label": canonical_for_label(label, canonical_by_label),
                "detector_score": float(score),
                "sources": [source],
            })
        batches.append({"batch": batch_index // labels_per_batch + 1, "labels": len(batch_labels), "detections": len(boxes), "seconds": round(perf_counter() - started, 2)})
    return detections, batches


def run_grounding_dino(image: Image.Image, labels: list[str], canonical_by_label: dict[str, str], device: str, box_threshold: float, text_threshold: float, labels_per_batch: int) -> tuple[list[dict], list[dict]]:
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    detections, batches = _grounding_dino_detections(
        image, labels, canonical_by_label, device, box_threshold, text_threshold, labels_per_batch,
        processor, model, "grounding_dino",
    )
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return nms_detections(detections), batches


def tile_regions(image: Image.Image, tile_size: int, overlap_ratio: float) -> list[tuple[int, int, int, int]]:
    """右端・下端を必ず含む、重なり付きタイルの座標を返す。"""
    if tile_size <= 0:
        raise ValueError("tile_size は正の整数にしてください。")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio は0以上1未満にしてください。")
    if max(image.size) <= tile_size:
        return [(0, 0, image.width, image.height)]
    stride = max(1, round(tile_size * (1 - overlap_ratio)))

    def starts(length: int) -> list[int]:
        final = max(0, length - tile_size)
        values = list(range(0, final + 1, stride))
        if not values or values[-1] != final:
            values.append(final)
        return values

    return [(x, y, min(x + tile_size, image.width), min(y + tile_size, image.height)) for y in starts(image.height) for x in starts(image.width)]


def run_tiled_grounding_dino(
    image: Image.Image,
    labels: list[str],
    canonical_by_label: dict[str, str],
    device: str,
    box_threshold: float,
    text_threshold: float,
    labels_per_batch: int,
    tile_size: int,
    overlap_ratio: float,
) -> tuple[list[dict], list[dict]]:
    """高解像度画像を重なり付きタイルで検出し、画像全体の座標へ戻して統合する。"""
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    detections, metrics = [], []
    for tile_index, (x1, y1, x2, y2) in enumerate(tile_regions(image, tile_size, overlap_ratio), start=1):
        tile = image.crop((x1, y1, x2, y2))
        tile_detections, batches = _grounding_dino_detections(
            tile, labels, canonical_by_label, device, box_threshold, text_threshold, labels_per_batch,
            processor, model, "grounding_dino_tile", offset=(x1, y1),
        )
        for detection in tile_detections:
            detection["tile_id"] = f"tile-{tile_index:03d}"
        detections.extend(tile_detections)
        metrics.append({"tile_id": f"tile-{tile_index:03d}", "origin": [x1, y1], "size": [x2 - x1, y2 - y1], "detections_before_merge": len(tile_detections), "seconds": round(sum(batch["seconds"] for batch in batches), 2)})
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merged = nms_detections(detections, threshold=0.50)
    return merged, metrics


def _as_mask_array(value: object) -> np.ndarray:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return np.squeeze(array).astype(bool)


def run_prompted_masks(image: Image.Image, detections: list[dict], device: str, batch_size: int) -> list[dict]:
    model_id = "facebook/sam2.1-hiera-small"
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device).eval()
    candidates = []
    for offset in range(0, len(detections), batch_size):
        group = detections[offset : offset + batch_size]
        boxes = [[item["bbox"] for item in group]]
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        processed = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())
        masks = processed[0]
        iou_scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
        for item, mask, score in zip(group, masks, iou_scores):
            candidate = dict(item)
            candidate.update({"candidate_id": f"obj-{len(candidates) + 1:04d}", "mask": _as_mask_array(mask), "mask_quality_score": float(score), "status": "pending", "validation": None})
            candidates.append(candidate)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return candidates


def run_automatic_masks(image: Image.Image, device: str, points_per_batch: int, score_threshold: float, min_area_ratio: float, max_area_ratio: float) -> list[dict]:
    device_index = 0 if device == "cuda" else -1
    generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-small", device=device_index)
    output = generator(image, points_per_batch=points_per_batch, pred_iou_thresh=score_threshold)
    image_area = image.width * image.height
    masks = []
    for mask, score in zip(output["masks"], output["scores"]):
        array = _as_mask_array(mask)
        ratio = array.sum() / image_area
        if min_area_ratio <= ratio <= max_area_ratio:
            ys, xs = np.where(array)
            masks.append({"mask": array, "mask_quality_score": float(score), "bbox": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], "area": int(array.sum())})
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return masks


def fuse_candidates(prompted: list[dict], automatic: list[dict], auto_match_iou: float, max_unmatched: int) -> tuple[list[dict], list[dict]]:
    unmatched = []
    for auto in automatic:
        best = max(prompted, key=lambda item: mask_iou(auto["mask"], item["mask"]), default=None)
        if best and mask_iou(auto["mask"], best["mask"]) >= auto_match_iou:
            best["sources"] = list(dict.fromkeys([*best["sources"], "sam2_auto"]))
        else:
            unmatched.append(auto)
    unmatched.sort(key=lambda item: item["mask_quality_score"], reverse=True)
    for item in unmatched[:max_unmatched]:
        prompted.append({"candidate_id": f"obj-{len(prompted) + 1:04d}", "bbox": item["bbox"], "mask": item["mask"], "mask_quality_score": item["mask_quality_score"], "detector_label": None, "canonical_label": None, "detector_score": 0.0, "sources": ["sam2_auto"], "status": "pending", "validation": None})
    return prompted, unmatched[max_unmatched:]


def mark_ambiguity(candidates: list[dict], detector_threshold: float, mask_threshold: float, mask_iou_threshold: float, containment_threshold: float) -> None:
    for candidate in candidates:
        reasons = []
        if candidate["detector_score"] < detector_threshold:
            reasons.append("LOW_DETECTOR_SCORE")
        if candidate["mask_quality_score"] < mask_threshold:
            reasons.append("LOW_MASK_QUALITY")
        if "sam2_auto" in candidate["sources"] and not any(source.startswith("grounding_dino") for source in candidate["sources"]):
            reasons.append("UNMATCHED_AUTO_MASK")
        for other in candidates:
            if other is candidate:
                continue
            if containment_ratio(candidate["mask"], other["mask"]) >= containment_threshold and candidate["mask"].sum() < other["mask"].sum():
                reasons.append("POSSIBLE_PART")
            if mask_iou(candidate["mask"], other["mask"]) >= mask_iou_threshold and candidate["canonical_label"] != other["canonical_label"]:
                reasons.append("LABEL_CONFLICT")
        candidate["ambiguity_reasons"] = sorted(set(reasons))


def candidate_images(image: Image.Image, candidate: dict) -> tuple[Image.Image, Image.Image]:
    array = np.asarray(image)
    mask = candidate["mask"]
    focus = array.copy()
    focus[~mask] = (focus[~mask] * 0.18).astype(np.uint8)
    x1, y1, x2, y2 = map(int, candidate["bbox"])
    padding = max(20, int(max(x2 - x1, y2 - y1) * 0.2))
    crop = image.crop((max(0, x1 - padding), max(0, y1 - padding), min(image.width, x2 + padding), min(image.height, y2 + padding)))
    return Image.fromarray(focus), crop


def validate_candidate(client: OpenAI, model_name: str, image: Image.Image, candidate: dict) -> tuple[dict | None, int]:
    prompt = f"""
候補ID {candidate['candidate_id']} を判定してください。1枚目は候補マスクを強調した全体画像、2枚目は周辺クロップです。
画像内の実物で独立して数える物体か、部品か、背景か、写真・絵・画面内に描かれた物体かを判定してください。
説明やMarkdownを含めず、必ず次のJSONだけを返してください。
{{"is_physical_object":true,"is_independent_instance":true,"is_part":false,"is_background":false,"is_depicted_object":false,"canonical_label":"mug","display_name_ja":"マグカップ","confidence":0.9,"reason_code":"VISIBLE_INDEPENDENT_OBJECT"}}
""".strip()
    focus, crop = candidate_images(image, candidate)
    attempts = 0
    for attempt in range(2):
        attempts += 1
        try:
            return call_vlm_json(client, model_name, prompt if attempt == 0 else prompt + " JSON形式を厳守してください。", [focus, crop]), attempts
        except (ValueError, json.JSONDecodeError):
            continue
        except Exception:
            return None, attempts
    return None, attempts


def apply_validation(candidates: list[dict], image: Image.Image, client: OpenAI, model_name: str, canonical_by_label: dict[str, str] | None = None, progress_callback=None) -> dict:
    """追加確認の進捗と、実際に行ったVLM呼び出し数を返す。"""
    targets = [candidate for candidate in candidates if candidate.get("ambiguity_reasons")]
    api_calls = 0
    completed = 0
    for candidate in candidates:
        if not candidate.get("ambiguity_reasons"):
            continue
        result, attempts = validate_candidate(client, model_name, image, candidate)
        api_calls += attempts
        completed += 1
        candidate["validation_attempts"] = attempts
        candidate["validation"] = result
        if result is None:
            candidate["status"] = "unknown" if candidate["canonical_label"] is None else "uncertain"
        else:
            returned_label = str(result.get("canonical_label") or candidate["canonical_label"] or "unknown").strip().lower()
            candidate["canonical_label"] = canonical_for_label(returned_label, canonical_by_label or {})
            candidate["display_name_ja"] = " ".join(str(result.get("display_name_ja") or candidate.get("display_name_ja") or "").split())
            excluded = any(result.get(key) for key in ("is_part", "is_background", "is_depicted_object")) or not result.get("is_physical_object", False) or not result.get("is_independent_instance", False)
            candidate["status"] = "rejected" if excluded else "pending"
        if progress_callback:
            progress_callback(completed, len(targets), api_calls, candidate)
    return {"targets": len(targets), "api_calls": api_calls, "completed": completed}


def finalise(candidates: list[dict], mask_iou_threshold: float) -> list[dict]:
    kept = []
    for candidate in sorted(candidates, key=lambda item: (item["detector_score"], item["mask_quality_score"]), reverse=True):
        if candidate["status"] == "rejected":
            continue
        duplicate = next((other for other in kept if candidate["canonical_label"] == other["canonical_label"] and mask_iou(candidate["mask"], other["mask"]) >= mask_iou_threshold), None)
        if duplicate:
            duplicate["sources"] = list(dict.fromkeys([*duplicate["sources"], *candidate["sources"]]))
            continue
        validation = candidate.get("validation")
        if validation:
            confidence = 0.35 * candidate["detector_score"] + 0.25 * candidate["mask_quality_score"] + 0.40 * float(validation.get("confidence", 0.0))
        else:
            confidence = 0.60 * candidate["detector_score"] + 0.40 * candidate["mask_quality_score"]
        candidate["final_confidence"] = float(confidence)
        if candidate["status"] == "unknown" or candidate["canonical_label"] == "unknown":
            candidate["status"] = "unknown"
        elif confidence >= 0.90:
            candidate["status"] = "confirmed"
        elif confidence >= 0.70:
            candidate["status"] = "probable"
        elif confidence >= 0.40:
            candidate["status"] = "uncertain"
        else:
            candidate["status"] = "rejected"
        kept.append(candidate)
    return kept


def candidate_color(candidate_id: str) -> tuple[int, int, int]:
    value = sum((index + 1) * ord(char) for index, char in enumerate(candidate_id))
    return ((value * 53) % 180 + 50, (value * 97) % 180 + 50, (value * 193) % 180 + 50)


def show_overlay(image: Image.Image, candidates: list[dict], title: str, show_masks: bool = True) -> None:
    frame = np.asarray(image).copy()
    labels = []
    for candidate in candidates:
        color = candidate_color(candidate["candidate_id"])
        if show_masks and "mask" in candidate:
            frame[candidate["mask"]] = (0.6 * frame[candidate["mask"]] + 0.4 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = map(int, candidate["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        labels.append((f"{candidate['candidate_id']} {display_label(candidate)}", x1, max(2, y1 - 20), color))
    canvas = Image.fromarray(frame)
    drawer = ImageDraw.Draw(canvas)
    font = japanese_font(16)
    for label, x, y, color in labels:
        drawer.text((x, y), label, fill=tuple(color), font=font, stroke_width=1, stroke_fill="black")
    plt.figure(figsize=(14, 10))
    plt.imshow(canvas)
    plt.title(title)
    plt.axis("off")
    plt.show()
    plt.close()


def show_gallery(image: Image.Image, candidates: list[dict], title: str, maximum: int = 30) -> None:
    selected = candidates[:maximum]
    if not selected:
        print(f"{title}: 対象なし")
        return
    columns = 5
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 3.5 * rows))
    for axis in np.ravel(axes):
        axis.axis("off")
    for axis, candidate in zip(np.ravel(axes), selected):
        _, crop = candidate_images(image, candidate)
        axis.imshow(crop)
        reasons = display_review_reasons(candidate.get("ambiguity_reasons", [])) or display_status(candidate.get("status", "pending"))
        axis.set_title(f"{candidate['candidate_id']}\n{display_label(candidate)}\n{reasons}", fontsize=8)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def candidate_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for item in candidates:
        rows.append({
            "候補ID": item["candidate_id"],
            "名前": display_label(item) if item.get("canonical_label") else "名前未確定",
            "場所探しAIの自信度": round(item.get("detector_score", 0.0), 3),
            "輪郭の自信度": round(item.get("mask_quality_score", 0.0), 3),
            "見つけた方法": ", ".join(item["sources"]),
            "追加確認の理由": display_review_reasons(item.get("ambiguity_reasons", [])),
            "判定": display_status(item.get("status", "pending")),
            "総合の自信度": round(item.get("final_confidence", 0.0), 3) if "final_confidence" in item else None,
        })
    return pd.DataFrame(rows)


def build_result(image: Image.Image, candidates: list[dict], configuration: dict) -> dict:
    categories = defaultdict(list)
    unknown = []
    for item in candidates:
        record = {"instance_id": item["candidate_id"], "bbox": [round(value, 1) for value in item["bbox"]], "confidence": round(item.get("final_confidence", 0.0), 3), "status": item["status"], "sources": item["sources"], "mask_area": int(item["mask"].sum()), "display_name_ja": item.get("display_name_ja") or None}
        if item["status"] == "unknown":
            unknown.append(record)
        elif item["status"] != "rejected":
            categories[item["canonical_label"]].append(record)
    category_rows = []
    for label, instances in sorted(categories.items()):
        states = pd.Series([item["status"] for item in instances]).value_counts()
        display_name_ja = next((item.get("display_name_ja") for item in instances if item.get("display_name_ja")), None)
        category_rows.append({"canonical_label": label, "display_name_ja": display_name_ja, "display_name": f"{label}（{display_name_ja}）" if display_name_ja else label, "confirmed_count": int(states.get("confirmed", 0)), "probable_count": int(states.get("probable", 0)), "uncertain_count": int(states.get("uncertain", 0)), "reported_count": int(states.get("confirmed", 0) + states.get("probable", 0)), "instances": instances})
    return {"status": "completed", "image": {"width": image.width, "height": image.height}, "summary": {"category_count": len(category_rows), "reported_instance_count": sum(row["reported_count"] for row in category_rows), "uncertain_instance_count": sum(row["uncertain_count"] for row in category_rows)}, "categories": category_rows, "unknown_objects": unknown, "configuration": configuration}
