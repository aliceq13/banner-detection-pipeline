"""
05_demo.py - 파이프라인 데모 및 전체 평가
==========================================
이 스크립트가 하는 일:
  1. COCO JSON(`config.COCO_BANNER_LBL`)에서 종류 라벨이 있는 GT 샘플 로드
  2. 각 샘플에 대해 YOLO seg → GT bbox와 IoU≥0.3 매칭 → 매칭된 폴리곤으로
     H-matrix 보정. 매칭 실패 시 GT bbox crop으로 fallback (`seg_corrected_from_gt`).
  3. 보정된 이미지를 VLM(`config.VLM_CONFIG["model_id"]`, 기본 Qwen3.5-9B)으로 분류해
     accuracy / precision / recall / F1 / Confusion Matrix 산출.
     첫 샘플은 classify(debug=True) 로 호출 — 이미지 토큰 / pixel_values
     shape 가 실제로 모델에 주입되는지 한 번 진단 로그로 확인한다.
  4. 카테고리 균등 샘플링한 발표용 grid figure 생성 (원본+GT bbox / 보정 / 분류).
  5. evaluation_report.md 요약 출력.

VRAM 스왑: YOLO로 모든 seg·보정을 선행 계산 → YOLO unload →
Qwen3.5-9B BF16(~18GB) 로드 → 평가. A5000 24GB에 동시 상주를 피해 안정.

플롯 라벨은 한글 글리프 누락 방지를 위해 전부 영문(KR_TO_EN). 내부 로깅·데이터는
한글 유지. 매 단계 try/except 로 감싸 일부 샘플 실패가 전체를 중단시키지 않게 함.

실행 방법:
  python 05_demo.py                        # 전체 평가 (n=50)
  python 05_demo.py --n-samples 10         # 10개만 빠르게 테스트
  python 05_demo.py --skip-vlm             # VLM 없이 탐지만 시연
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config

# matplotlib / cv2 폰트가 한글을 렌더하지 못해 glyph missing 경고가 나므로
# plot 라벨은 전부 영문으로 고정한다. (내부 데이터·로그는 한글 유지)
KR_TO_EN = {
    "정당 현수막": "Party",
    "민간 현수막": "Private",
    "공공 현수막": "Public",
}


def parse_args():
    parser = argparse.ArgumentParser(description="현수막 파이프라인 데모 및 평가")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="평가할 샘플 수 (기본: 50)")
    parser.add_argument("--skip-vlm", action="store_true",
                        help="VLM 분류 건너뜀 (빠른 데모용)")
    parser.add_argument("--model", default=str(config.YOLO_BEST_MODEL),
                        help="YOLO 모델 경로")
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "demo"),
                        help="결과 저장 디렉토리")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


# ============================================================
# VLM 분류 정확도 평가
# ============================================================

def load_gt_labels(n_samples: int, seed: int) -> List[Tuple[Path, str]]:
    """
    temp 데이터에서 ground truth 레이블이 있는 샘플을 로드합니다.

    각 COCO JSON에서 현수막 종류(정당/민간/공공) 정보를 추출하고
    해당 현수막 영역을 crop하여 평가에 사용합니다.

    Args:
        n_samples: 샘플 수
        seed: 랜덤 시드

    Returns:
        [(이미지_경로, 정답_카테고리), ...]
    """
    print("[GT] ground truth 샘플 로드 중...")

    label_dir = config.COCO_BANNER_LBL
    image_dir  = config.COCO_BANNER_IMG

    samples = []

    category_map = {
        "정당 현수막": "정당 현수막",
        "민간 현수막": "민간 현수막",
        "공공 현수막": "공공 현수막",
    }

    for json_file in label_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        cat_map = {c['id']: c['name'] for c in data.get('categories', [])}
        img_map = {img['id']: img['file_name'] for img in data.get('images', [])}

        for ann in data.get('annotations', []):
            cat_name = cat_map.get(ann['category_id'], '')
            if cat_name not in category_map:
                continue

            img_file = image_dir / img_map.get(ann['image_id'], '')
            if not img_file.exists():
                continue

            bbox = ann.get('bbox', [])
            if len(bbox) < 4 or bbox[2] < 30 or bbox[3] < 30:
                continue

            samples.append((img_file, bbox, category_map[cat_name]))

    # 랜덤 샘플링
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:n_samples]
    print(f"[GT] {len(samples)}개 샘플 로드 완료")

    # 카테고리별 통계
    from collections import Counter
    cat_counts = Counter(s[2] for s in samples)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}개")

    return samples


def crop_banner_from_image(img_path: Path, bbox: list) -> np.ndarray:
    """GT bbox([x,y,w,h])로 이미지를 잘라 BGR crop을 반환한다. 실패 시 None."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    x, y, w, h = [int(v) for v in bbox]
    ih, iw = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)

    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _bbox_iou_xywh_xyxy(gt_xywh, det_xyxy) -> float:
    """GT bbox([x,y,w,h])와 YOLO 감지 bbox([x1,y1,x2,y2])의 IoU."""
    gx, gy, gw, gh = gt_xywh
    gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
    dx1, dy1, dx2, dy2 = det_xyxy
    ix1, iy1 = max(gx1, dx1), max(gy1, dy1)
    ix2, iy2 = min(gx2, dx2), min(gy2, dy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = gw * gh + (dx2 - dx1) * (dy2 - dy1) - inter
    return inter / union if union > 0 else 0.0


def seg_corrected_from_gt(
    pipeline, img_path: Path, gt_bbox: list, iou_thr: float = 0.3,
):
    """GT bbox와 가장 겹치는 YOLO seg 감지의 polygon으로 H-matrix 보정.

    Returns:
        (corrected_image, matched_detection, iou) 또는 (None, None, 0.0) 실패 시.
        감지가 아예 없거나 매칭이 안 되면 None을 반환 — 호출자가 fallback 처리.
    """
    image = cv2.imread(str(img_path))
    if image is None:
        return None, None, 0.0

    detections = pipeline.detector.detect(image)
    banners = [d for d in detections if d['class_id'] == 0]
    if not banners:
        return None, None, 0.0

    best, best_iou = None, 0.0
    for d in banners:
        iou = _bbox_iou_xywh_xyxy(gt_bbox, d['bbox'])
        if iou > best_iou:
            best, best_iou = d, iou
    if best is None or best_iou < iou_thr:
        return None, None, best_iou

    corrected, _ = pipeline.corrector.correct(image, best['polygon'])
    return corrected, best, best_iou


def precompute_corrections(pipeline, samples: list) -> list:
    """모든 샘플에 대해 YOLO seg 감지 + H-matrix 보정을 선행 수행하여 캐싱한다.

    VLM(Qwen3.5-9B, BF16 ~18GB)을 로드하기 전에 YOLO를 VRAM에서 내리기 위해,
    필요한 모든 YOLO 결과(보정 이미지·IoU·fallback 여부)를 미리 뽑아 둔다.

    Returns:
        [{'img_path', 'bbox', 'gt_cat', 'corrected', 'iou', 'fallback'}] 리스트.
        corrected가 None이면 해당 샘플은 이후 분류에서 제외된다.
    """
    from tqdm import tqdm
    print(f"\n[Precompute] YOLO detect + H-matrix on {len(samples)} samples...")
    cached = []
    for img_path, bbox, gt_cat in tqdm(samples, desc="YOLO+warp"):
        try:
            corrected, _matched, iou = seg_corrected_from_gt(pipeline, img_path, bbox)
        except Exception as e:
            print(f"  [WARN] seg/warp error ({img_path.name}): {e}")
            corrected, iou = None, 0.0

        fallback = False
        if corrected is None:
            crop = crop_banner_from_image(img_path, bbox)
            if crop is not None and crop.size > 0:
                corrected = crop
                fallback = True

        cached.append({
            'img_path':  img_path,
            'bbox':      bbox,
            'gt_cat':    gt_cat,
            'corrected': corrected,
            'iou':       iou,
            'fallback':  fallback,
        })
    n_seg  = sum(1 for c in cached if c['corrected'] is not None and not c['fallback'])
    n_crop = sum(1 for c in cached if c['fallback'])
    n_fail = sum(1 for c in cached if c['corrected'] is None)
    print(f"[Precompute] seg match: {n_seg}, crop fallback: {n_crop}, failed: {n_fail}")
    return cached


def evaluate_vlm_classification(
    pipeline,
    cached: list,
    output_dir: Path,
) -> dict:
    """
    사전 계산된 보정 이미지에 VLM 분류를 적용해 정확도를 평가한다.

    평가 지표:
    - 전체 정확도 (Accuracy)
    - 클래스별 Precision, Recall, F1
    - Confusion Matrix

    Args:
        pipeline: BannerPipeline (classifier 만 필요, detector는 이미 해제돼도 OK)
        cached:   precompute_corrections()가 반환한 리스트
        output_dir: 결과 저장 디렉토리

    Returns:
        평가 결과 딕셔너리
    """
    print("\n[VLM Eval] measuring classification accuracy...")
    from tqdm import tqdm
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix
    )
    import seaborn as sns

    y_true, y_pred = [], []
    n_seg_match = sum(1 for c in cached if c['corrected'] is not None and not c['fallback'])
    n_crop_fallback = sum(1 for c in cached if c['fallback'])
    n_failed = sum(1 for c in cached if c['corrected'] is None)
    categories = ["정당 현수막", "민간 현수막", "공공 현수막"]

    first_sample_debugged = False
    for c in tqdm(cached, desc="VLM classify"):
        if c['corrected'] is None:
            continue

        try:
            # 첫 평가 샘플만 debug=True — 이미지 토큰/픽셀 shape이 실제로
            # 주입되는지 한 번만 확인하고 나머지는 조용히 진행.
            debug = not first_sample_debugged
            result = pipeline.classifier.classify(c['corrected'], debug=debug)
            first_sample_debugged = True
            pred_cat = result['category']
        except Exception as e:
            print(f"  [WARN] classify error ({c['img_path'].name}): {e}")
            n_failed += 1
            continue

        y_true.append(c['gt_cat'])
        y_pred.append(pred_cat)

    print(f"\n  seg match: {n_seg_match}, crop fallback: {n_crop_fallback}, "
          f"failed: {n_failed}")
    print(f"  evaluated: {len(y_true)}")

    if not y_true:
        print("[ERROR] No evaluable samples")
        return {
            'accuracy': 0.0, 'report': {}, 'confusion': None,
            'n_evaluated': 0, 'n_failed': n_failed,
            'n_seg_match': n_seg_match, 'n_crop_fallback': n_crop_fallback,
        }

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=categories,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=categories)

    print(f"\n[VLM Result]")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print()
    print(classification_report(y_true, y_pred, labels=categories, zero_division=0))

    # ── Confusion Matrix 시각화 ──────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    en_labels = [KR_TO_EN.get(c, c) for c in categories]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=en_labels,
        yticklabels=en_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"VLM Classification Confusion Matrix (Accuracy: {acc:.3f})", fontsize=13)
    plt.tight_layout()
    cm_path = output_dir / "vlm_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVE] Confusion Matrix: {cm_path}")

    return {
        'accuracy':        acc,
        'report':          report,
        'confusion':       cm,
        'n_evaluated':     len(y_true),
        'n_failed':        n_failed,
        'n_seg_match':     n_seg_match,
        'n_crop_fallback': n_crop_fallback,
    }


# ============================================================
# 발표용 시각화
# ============================================================

def _balance_cached_by_category(
    cached: list, n_display: int, categories: List[str],
) -> list:
    """발표용 figure 에 쓸 '좋은' 샘플을 카테고리별 균등하게 뽑는다.

    우선순위 (내림차순):
    1. seg 매칭 성공 (fallback=False, H-matrix 보정이 실제 적용된 샘플)
    2. IoU 가 높음 (YOLO 감지와 GT 가 잘 맞는 샘플)
    3. GT bbox 면적이 큼 (배너가 크게 나와 시각적으로 명확)

    각 카테고리에서 위 우선순위로 정렬한 뒤 per_cat = n_display//len(categories)
    개씩 선택. 실패 샘플(corrected=None)은 제외.
    """
    if not categories:
        return cached[:n_display]
    per_cat = max(1, n_display // len(categories))

    def score(c):
        bbox = c.get('bbox', [0, 0, 0, 0])
        area = float(bbox[2]) * float(bbox[3]) if len(bbox) >= 4 else 0.0
        # (fallback 아님 우선, IoU 높을수록 우선, 면적 큰 순)
        return (0 if c.get('fallback') else 1, float(c.get('iou', 0.0)), area)

    by_cat: dict = {cat: [] for cat in categories}
    for c in cached:
        if c.get('corrected') is None:
            continue
        cat = c['gt_cat']
        if cat in by_cat:
            by_cat[cat].append(c)

    selected = []
    for cat in categories:
        ranked = sorted(by_cat[cat], key=score, reverse=True)
        selected.extend(ranked[:per_cat])
    return selected[:n_display]


def create_presentation_figure(
    cached: list,
    pipeline,
    output_dir: Path,
    n_display: int = 6,
):
    """발표용 파이프라인 시연 이미지 (원본 / H-matrix 보정 / VLM 분류)를 생성한다.

    구조: N행 × 3열 grid. 각 행은 하나의 샘플.
    precompute_corrections() 결과(cached)를 사용하므로 YOLO는 이미 해제돼 있어도 OK.
    모든 라벨은 영문 (한글 폰트 미지원 시 glyph 경고 방지).
    """
    print(f"\n[Figure] Rendering {n_display} samples...")
    from tqdm import tqdm

    categories = ["정당 현수막", "민간 현수막", "공공 현수막"]
    selected = _balance_cached_by_category(cached, n_display, categories)
    if not selected:
        print("[WARN] No samples to display")
        return

    fig = plt.figure(figsize=(16, 4 * len(selected)))
    fig.suptitle("Banner Pipeline: YOLO26-seg -> H-Matrix -> Qwen3.5 VLM",
                 fontsize=18, fontweight='bold')

    for row_idx, c in enumerate(tqdm(selected, desc="Figure")):
        img_path = c['img_path']
        bbox     = c['bbox']
        gt_cat   = c['gt_cat']
        corrected     = c['corrected']
        iou           = c['iou']
        fallback_used = c['fallback']

        img = cv2.imread(str(img_path))
        gt_en = KR_TO_EN.get(gt_cat, gt_cat)

        ax1 = fig.add_subplot(len(selected), 3, row_idx * 3 + 1)
        ax2 = fig.add_subplot(len(selected), 3, row_idx * 3 + 2)
        ax3 = fig.add_subplot(len(selected), 3, row_idx * 3 + 3)
        for ax in (ax1, ax2, ax3):
            ax.axis('off')

        if img is None:
            ax1.text(0.5, 0.5, "Image load failed",
                     ha='center', va='center', transform=ax1.transAxes)
            continue

        # (1) 원본 + GT bbox
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x, y, w, h = [int(v) for v in bbox]
        ax1.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
        ax1.set_title(f"Original (GT: {gt_en})", fontsize=13)

        # (2) 사전 계산된 보정 이미지
        if corrected is not None and corrected.size > 0:
            ax2.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            ax2.set_title(
                "GT crop (seg match failed)" if fallback_used
                else f"H-Matrix warp (IoU={iou:.2f})",
                fontsize=13,
            )
        else:
            ax2.text(0.5, 0.5, "Crop failed",
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("H-Matrix warp", fontsize=13)

        # (3) VLM 분류
        ax3.set_title("VLM classify", fontsize=13)
        if pipeline.classifier is None:
            ax3.text(0.5, 0.5, "VLM disabled",
                     ha='center', va='center', transform=ax3.transAxes)
            continue
        if corrected is None or corrected.size == 0:
            ax3.text(0.5, 0.5, "No input",
                     ha='center', va='center', transform=ax3.transAxes)
            continue

        try:
            cls_result = pipeline.classifier.classify(corrected)
        except Exception as e:
            ax3.text(0.5, 0.5, f"Classify failed\n{str(e)[:60]}",
                     ha='center', va='center', transform=ax3.transAxes,
                     fontsize=9)
            continue

        pred_en = KR_TO_EN.get(cls_result['category'], cls_result['category'])
        ok = cls_result['category'] == gt_cat
        color = 'green' if ok else 'red'
        # 한글 figure rendering은 glyph 누락이 나므로 Korean text는 숨기고
        # 영문 라벨 + 신뢰도·OK/X 만 표시. 추출된 문구는 로그/report 에서 확인.
        text = (
            f"Pred: {pred_en}\n"
            f"Conf: {cls_result['confidence']:.2f}\n"
            f"GT:   {gt_en}\n"
            f"{'[OK]' if ok else '[X]'}"
        )
        ax3.text(0.5, 0.5, text, ha='center', va='center',
                 transform=ax3.transAxes,
                 fontsize=20, fontweight='bold', color=color, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                           edgecolor=color, linewidth=2))

    plt.tight_layout()
    save_path = output_dir / "pipeline_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVE] Pipeline demo figure: {save_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Banner Pipeline Demo & Evaluation")
    print("=" * 60)

    if not Path(args.model).exists():
        print(f"[ERROR] YOLO 가중치가 없습니다: {args.model}")
        print("  학습된 가중치 다운로드:")
        print("    mkdir -p results/models && \\")
        print("    curl -L -o results/models/yolo26_banner_best.pt \\")
        print("      https://github.com/aliceq13/banner-detection-pipeline/releases/download/v1.0/binary2-best.pt")
        sys.exit(1)

    if not config.COCO_BANNER_LBL.exists():
        print(f"[ERROR] GT 레이블 디렉토리가 없습니다: {config.COCO_BANNER_LBL}")
        print("  05_demo.py 는 원천 `illegal_banner` 데이터셋(COCO JSON 레이블)이 필요합니다.")
        print("  docker-compose.yml 의 `/workspace/illegal_banner` 볼륨 주석을 해제하고")
        print("  실제 데이터셋 경로를 지정한 뒤 `docker compose up -d --force-recreate` 하세요.")
        print("  데이터셋이 없다면 04_pipeline.py --image <path> 로 단일 이미지 추론을 사용하세요.")
        sys.exit(1)

    from pipeline import BannerPipeline
    # 주의: 여기서는 VLM을 로드하지 않는다. 먼저 YOLO로 모든 감지·보정을
    # 사전 수행한 뒤 YOLO를 VRAM에서 내리고, 그 자리에 VLM을 올려 Qwen3.5-9B
    # BF16(~18GB) 과 YOLO 가 경쟁하지 않도록 VRAM 스왑한다.
    try:
        pipeline = BannerPipeline(
            yolo_model_path=args.model,
            load_vlm=False,
        )
    except Exception as e:
        print(f"[FATAL] Pipeline init failed: {e}")
        sys.exit(1)

    gt_samples = load_gt_labels(args.n_samples, args.seed)
    if not gt_samples:
        print(f"[ERROR] GT 샘플을 불러오지 못했습니다 (label dir: {config.COCO_BANNER_LBL}).")
        print("  JSON 내부에 categories/annotations 가 비어 있거나 이미지 경로가 어긋났을 수 있습니다.")
        sys.exit(1)

    # 1. YOLO 선행 실행 (모든 샘플 보정 이미지 캐싱)
    try:
        cached = precompute_corrections(pipeline, gt_samples)
    except Exception as e:
        print(f"[FATAL] Precompute failed: {e}")
        sys.exit(1)

    # 2. YOLO 해제 후 VLM 로드 (VRAM 확보)
    if not args.skip_vlm:
        print("\n[Swap] Unloading YOLO, loading VLM...")
        pipeline.unload_detector()
        pipeline.load_classifier()
    else:
        print("[INFO] --skip-vlm: VLM not loaded")

    # 3. VLM 평가
    eval_results = {}
    if pipeline.classifier:
        try:
            eval_results = evaluate_vlm_classification(
                pipeline=pipeline,
                cached=cached,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"[ERROR] VLM evaluation crashed: {e}")
    else:
        print("[INFO] VLM disabled -> skipping classification eval")

    # 4. 발표용 figure (cached 사용 — YOLO 불필요)
    try:
        create_presentation_figure(
            cached=cached,
            pipeline=pipeline,
            output_dir=output_dir,
            n_display=6,
        )
    except Exception as e:
        print(f"[ERROR] Presentation figure failed: {e}")

    _write_summary_report(eval_results, output_dir)

    print(f"\n[DONE] All results saved to: {output_dir}")


def _write_summary_report(eval_results: dict, output_dir: Path):
    """평가 결과 요약(Markdown)을 output_dir/evaluation_report.md 에 쓴다."""
    report_path = output_dir / "evaluation_report.md"
    vlm_id = config.VLM_CONFIG["model_id"]
    lines = [
        "# 현수막 탐지 파이프라인 평가 보고서",
        "",
        "## 파이프라인 구성",
        "| 단계 | 모델/방식 | 역할 |",
        "|------|----------|------|",
        "| 1. 탐지 | YOLO26-seg | 현수막 영역 세그멘테이션 |",
        "| 2. 보정 | H-Matrix (4 extreme corners + getPerspectiveTransform) | 원근 왜곡 보정 |",
        f"| 3. 분류 | {vlm_id} | 현수막 종류 분류 |",
        "",
    ]

    if eval_results:
        acc = eval_results.get('accuracy', 0.0)
        n_eval = eval_results.get('n_evaluated', 0)
        n_seg = eval_results.get('n_seg_match', 0)
        n_crop = eval_results.get('n_crop_fallback', 0)
        n_fail = eval_results.get('n_failed', 0)
        lines += [
            "## VLM 분류 결과",
            f"- Accuracy: **{acc:.4f} ({acc*100:.1f}%)**",
            f"- 평가 샘플: {n_eval}개 (seg 매칭 {n_seg} / crop fallback {n_crop} / 실패 {n_fail})",
            "",
            "## 클래스별 성능",
            "| 카테고리 | Precision | Recall | F1 |",
            "|----------|-----------|--------|----|",
        ]
        report = eval_results.get('report', {})
        for cat in ["정당 현수막", "민간 현수막", "공공 현수막"]:
            r = report.get(cat)
            if r:
                lines.append(
                    f"| {cat} | {r['precision']:.3f} | "
                    f"{r['recall']:.3f} | {r['f1-score']:.3f} |"
                )

    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  [SAVE] Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
