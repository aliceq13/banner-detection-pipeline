"""
03_evaluate_yolo.py - YOLO26 모델 성능 평가
===========================================
이 스크립트가 하는 일:
  1. 학습된 YOLO26 모델로 test 세트 추론
  2. 평가 지표 계산: mAP50, mAP50-95, Precision, Recall
  3. Confusion Matrix 시각화
  4. 예측 결과 샘플 이미지 저장
  5. COCO negative vs 현수막 positive 성능 분석

평가 지표 설명:
  - mAP50: IoU=0.5 기준 평균 정밀도
    - IoU(Intersection over Union): 예측 마스크와 실제 마스크의 겹치는 비율
    - 일반적으로 mAP50 > 0.5 이면 실용적 수준
  - mAP50-95: IoU=0.5~0.95 평균 (더 엄격한 기준)
  - Precision: 탐지한 것 중 실제 현수막인 비율 (오검출 적을수록 높음)
  - Recall: 실제 현수막 중 탐지한 비율 (미검출 적을수록 높음)

실행 방법:
  python 03_evaluate_yolo.py
  python 03_evaluate_yolo.py --model path/to/custom.pt
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 서버에서 matplotlib 사용
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26 현수막 탐지 모델 평가"
    )
    parser.add_argument(
        "--model",
        default=str(config.YOLO_BEST_MODEL),
        help=f"평가할 모델 경로 (기본: {config.YOLO_BEST_MODEL})"
    )
    parser.add_argument(
        "--data",
        default=str(config.DATASET_YAML),
        help="dataset.yaml 경로"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="평가할 데이터 분할 (기본: test)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="신뢰도 임계값 (기본: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU 임계값 (기본: 0.6)"
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=20,
        help="저장할 샘플 이미지 수 (기본: 20)"
    )
    return parser.parse_args()


def evaluate_model(args):
    """
    YOLO 모델을 test 세트로 평가하고 결과를 출력합니다.

    평가 절차:
    1. 모델 로드
    2. test 세트에 대해 val() 실행 → 공식 mAP 계산
    3. 샘플 이미지에 예측 결과 시각화
    4. COCO negative vs positive 분리 분석

    Args:
        args: parse_args() 결과
    """
    print("=" * 60)
    print("YOLO26 모델 성능 평가")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 미설치")
        sys.exit(1)

    # ── 모델 로드 ─────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 모델 파일 없음: {model_path}")
        print("  → 02_train_yolo.py로 먼저 모델을 학습하세요")
        sys.exit(1)

    print(f"\n[INFO] 모델 로드: {model_path}")
    model = YOLO(str(model_path))

    # ── 공식 평가 실행 ────────────────────────────────────────
    print(f"\n[INFO] {args.split} 세트 평가 시작...")
    eval_save_dir = config.RESULTS_DIR / "evaluation"
    eval_save_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_json=True,    # COCO 포맷 JSON 결과 저장
        save_hybrid=True,  # 시각화 이미지 저장
        plots=True,
        project=str(config.RESULTS_DIR / "evaluation"),
        name=f"yolo_eval_{args.split}",
        verbose=True,
    )

    # ── 결과 출력 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)

    # Box detection 메트릭
    print("\n[Box Detection]")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.mp:.4f}")
    print(f"  Recall:   {metrics.box.mr:.4f}")

    # Segmentation mask 메트릭 (더 중요!)
    if hasattr(metrics, 'seg'):
        print("\n[Segmentation Mask]")
        print(f"  mAP50:    {metrics.seg.map50:.4f}")
        print(f"  mAP50-95: {metrics.seg.map:.4f}")
        print(f"  Precision:{metrics.seg.mp:.4f}")
        print(f"  Recall:   {metrics.seg.mr:.4f}")

    # 클래스별 성능
    print("\n[클래스별 mAP50]")
    class_names = config.YOLO_CLASSES
    if hasattr(metrics.box, 'ap_class_index'):
        for idx, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            ap50 = metrics.box.ap50[idx] if hasattr(metrics.box, 'ap50') else 0
            print(f"  {cls_name}: {ap50:.4f}")

    # ── 성능 시각화 그래프 저장 ──────────────────────────────
    _plot_metrics(metrics, eval_save_dir)

    # ── COCO negative vs Positive 분석 ──────────────────────
    _analyze_negative_vs_positive(
        model=model,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        n_samples=args.save_samples,
        save_dir=eval_save_dir,
    )

    print(f"\n[DONE] 평가 결과 저장: {eval_save_dir}")
    print(f"다음 단계: python 04_pipeline.py")

    return metrics


def _plot_metrics(metrics, save_dir: Path):
    """
    평가 결과를 막대 그래프로 시각화합니다.

    Args:
        metrics: YOLO val() 결과
        save_dir: 그래프 저장 디렉토리
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("YOLO26 Banner Detection 성능", fontsize=14)

        # Box metrics
        labels = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
        box_vals = [
            metrics.box.map50,
            metrics.box.map,
            metrics.box.mp,
            metrics.box.mr,
        ]

        ax = axes[0]
        bars = ax.bar(labels, box_vals, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        ax.set_title("Box Detection 메트릭")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, box_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10
            )

        # Seg metrics (있으면)
        if hasattr(metrics, 'seg'):
            seg_vals = [
                metrics.seg.map50,
                metrics.seg.map,
                metrics.seg.mp,
                metrics.seg.mr,
            ]
            ax2 = axes[1]
            bars2 = ax2.bar(labels, seg_vals, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
            ax2.set_title("Segmentation Mask 메트릭")
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("Score")
            for bar, val in zip(bars2, seg_vals):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10
                )
        else:
            axes[1].text(0.5, 0.5, "Segmentation\n메트릭 없음",
                        ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        save_path = save_dir / "evaluation_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [PLOT] 메트릭 그래프 저장: {save_path}")

    except Exception as e:
        print(f"  [WARN] 그래프 저장 실패: {e}")


def _analyze_negative_vs_positive(
    model,
    data_yaml: str,
    split: str,
    conf: float,
    n_samples: int,
    save_dir: Path,
):
    """
    COCO negative 이미지와 현수막 positive 이미지에 대한
    False Positive / True Positive 분석을 수행합니다.

    COCO negative에서 오검출(False Positive)이 많으면
    → 모델이 현수막이 아닌 것을 현수막으로 잘못 탐지하는 문제가 있음

    Args:
        model: 로드된 YOLO 모델
        data_yaml: 데이터셋 YAML 경로
        split: 평가할 split
        conf: 신뢰도 임계값
        n_samples: 분석할 샘플 수
        save_dir: 결과 저장 디렉토리
    """
    import yaml
    import cv2
    import random

    save_dir = Path(save_dir)

    try:
        with open(data_yaml, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return

    data_root = Path(cfg.get('path', ''))
    img_dir   = data_root / cfg.get(split, '')

    if not img_dir.exists():
        print(f"  [WARN] 이미지 경로 없음: {img_dir}")
        return

    all_images = list(img_dir.glob("*.jpg"))
    # COCO negative: 파일명이 "COCO_"로 시작
    coco_imgs    = [p for p in all_images if p.name.startswith("COCO_")]
    banner_imgs  = [p for p in all_images if not p.name.startswith("COCO_")]

    random.seed(42)
    sample_coco   = random.sample(coco_imgs,   min(n_samples, len(coco_imgs)))
    sample_banner = random.sample(banner_imgs, min(n_samples, len(banner_imgs)))

    fp_count = 0  # False Positive: COCO 이미지에서 현수막 탐지
    tp_count = 0  # True Positive:  현수막 이미지에서 현수막 탐지
    fn_count = 0  # False Negative: 현수막 이미지에서 현수막 미탐지

    # COCO 이미지 분석 (false positive 체크)
    for img_path in sample_coco:
        results = model(str(img_path), conf=conf, verbose=False)
        if len(results[0].boxes) > 0:
            fp_count += 1  # 현수막이 없는데 탐지됨

    # 현수막 이미지 분석 (true positive / false negative 체크)
    for img_path in sample_banner:
        results = model(str(img_path), conf=conf, verbose=False)
        if len(results[0].boxes) > 0:
            tp_count += 1   # 현수막 탐지 성공
        else:
            fn_count += 1   # 현수막 미탐지

    print("\n[COCO negative vs Banner positive 분석]")
    print(f"  COCO 샘플 {len(sample_coco)}장 중 오검출: {fp_count}장 "
          f"(FP rate: {fp_count/max(1,len(sample_coco)):.2%})")
    print(f"  Banner 샘플 {len(sample_banner)}장 중 탐지 성공: {tp_count}장 "
          f"(TP rate: {tp_count/max(1,len(sample_banner)):.2%})")
    print(f"  Banner 샘플 {len(sample_banner)}장 중 탐지 실패: {fn_count}장 "
          f"(FN rate: {fn_count/max(1,len(sample_banner)):.2%})")

    # ── 비교 시각화 저장 ─────────────────────────────────────
    _save_detection_samples(
        model=model,
        positive_imgs=sample_banner[:5],
        negative_imgs=sample_coco[:5],
        conf=conf,
        save_dir=save_dir / "detection_samples",
    )


def _save_detection_samples(
    model,
    positive_imgs: list,
    negative_imgs: list,
    conf: float,
    save_dir: Path,
):
    """
    탐지 결과 샘플 이미지를 저장합니다.
    발표 자료에 활용할 수 있습니다.

    Args:
        model: YOLO 모델
        positive_imgs: 현수막 있는 이미지 경로 목록
        negative_imgs: COCO 이미지 경로 목록 (현수막 없음)
        conf: 신뢰도 임계값
        save_dir: 저장 디렉토리
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [SAVE] 탐지 샘플 이미지 저장: {save_dir}")

    for label, imgs in [("positive", positive_imgs), ("negative", negative_imgs)]:
        for i, img_path in enumerate(imgs):
            try:
                results = model(str(img_path), conf=conf, verbose=False)
                # YOLO 탐지 결과가 그려진 이미지 저장
                annotated = results[0].plot()
                import cv2
                save_path = save_dir / f"{label}_{i:02d}_{img_path.stem}.jpg"
                cv2.imwrite(str(save_path), annotated)
            except Exception as e:
                print(f"    [WARN] 샘플 저장 실패 ({img_path.name}): {e}")


def main():
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
