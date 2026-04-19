"""
02_train_yolo.py - YOLO26 세그멘테이션 모델 학습
=================================================
이 스크립트가 하는 일:
  1. YOLO26 segmentation 모델 초기화 (사전학습 가중치 사용)
  2. 현수막 + COCO negative 데이터로 학습
  3. 학습 곡선, mAP 등 결과 저장
  4. 최적 모델 가중치를 결과 디렉토리에 복사

YOLO26 세그멘테이션이란?
  - YOLO (You Only Look Once): 실시간 객체 탐지 알고리즘
  - Segmentation: bounding box 대신 픽셀 단위 마스크 출력
  - 현수막의 불규칙한 형태를 정확히 표현하기 위해 segmentation 사용
  - 세그멘테이션 결과의 꼭짓점 → H-matrix 계산에 활용

학습 데이터:
  - Positive: banner(현수막) + text(텍스트) 어노테이션이 있는 이미지
  - Negative: COCO 이미지 (빈 레이블 = 아무 것도 없는 배경)

실행 방법:
  python 02_train_yolo.py
  python 02_train_yolo.py --epochs 50 --batch 16  # 빠른 테스트
"""

import sys
import shutil
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))
import config


def parse_args():
    """
    커맨드라인 인자를 파싱합니다.
    기본값은 config.py의 YOLO_CONFIG에서 가져옵니다.
    """
    parser = argparse.ArgumentParser(
        description="YOLO26 현수막 세그멘테이션 모델 학습"
    )
    parser.add_argument(
        "--model",
        default=config.YOLO_CONFIG["model"],
        help=f"YOLO 모델 (기본: {config.YOLO_CONFIG['model']})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.YOLO_CONFIG["epochs"],
        help=f"학습 에폭 수 (기본: {config.YOLO_CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=config.YOLO_CONFIG["batch"],
        help="배치 크기 (-1: 자동, 기본: -1)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.YOLO_CONFIG["imgsz"],
        help=f"입력 이미지 크기 (기본: {config.YOLO_CONFIG['imgsz']})"
    )
    parser.add_argument(
        "--device",
        default=str(config.YOLO_CONFIG["device"]),
        help="학습 디바이스 (기본: 0)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 학습 이어서 진행"
    )
    parser.add_argument(
        "--data",
        default=str(config.DATASET_YAML),
        help="dataset.yaml 경로"
    )
    parser.add_argument(
        "--name",
        default=config.YOLO_CONFIG["name"],
        help=f"결과 저장 폴더 이름 (기본: {config.YOLO_CONFIG['name']})"
    )
    return parser.parse_args()


def check_dataset(data_yaml: Path) -> bool:
    """
    학습 전 데이터셋 파일이 올바르게 준비되었는지 확인합니다.

    Args:
        data_yaml: dataset.yaml 경로

    Returns:
        True: 정상, False: 문제 있음
    """
    if not data_yaml.exists():
        print(f"[ERROR] dataset.yaml 없음: {data_yaml}")
        print("  → 먼저 01_prepare_data.py를 실행하세요")
        return False

    # YAML 파싱하여 경로 확인
    import yaml
    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get('path', ''))
    for split_key in ['train', 'val']:
        split_path = data_root / cfg.get(split_key, '')
        if not split_path.exists():
            print(f"[ERROR] 데이터 경로 없음: {split_path}")
            print("  → 01_prepare_data.py를 실행하여 데이터를 준비하세요")
            return False
        n_imgs = len(list(split_path.glob("*.jpg")))
        print(f"  [{split_key:5s}] {n_imgs}장 확인 완료")

    return True


def train(args):
    """
    YOLO26 세그멘테이션 모델을 학습합니다.

    학습 과정:
    1. 모델 로드: 사전학습된 YOLO26 가중치 로드
       - 사전학습이란? ImageNet 등 대규모 데이터로 미리 학습된 가중치
       - Transfer Learning: 좋은 특징 추출기에서 시작하므로 수렴 빠름
    2. Fine-tuning: 현수막 데이터로 특화 학습
       - 초기 에폭: 천천히 학습 (lr warmup)
       - 이후: 지정한 학습률로 학습
    3. 검증: 매 에폭마다 val 세트로 mAP 계산
    4. Early stopping: patience 에폭 동안 개선 없으면 학습 중단

    Args:
        args: parse_args() 결과
    """
    print("=" * 60)
    print("YOLO26 현수막 세그멘테이션 학습 시작")
    print("=" * 60)

    # ── 필수 패키지 import ────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 미설치. pip install ultralytics")
        sys.exit(1)

    # ── 데이터셋 확인 ─────────────────────────────────────────
    data_yaml = Path(args.data)
    print(f"\n[INFO] 데이터셋 확인: {data_yaml}")
    if not check_dataset(data_yaml):
        sys.exit(1)

    # ── 모델 로드 ─────────────────────────────────────────────
    print(f"\n[INFO] 모델 로드: {args.model}")
    print("  - 사전학습 가중치를 자동으로 다운로드합니다 (최초 실행 시)")
    print("  - 이후 실행시 캐시에서 로드됩니다")

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        print("  YOLO26 모델을 찾을 수 없습니다.")
        print("  ultralytics를 최신 버전으로 업데이트하세요: pip install -U ultralytics")
        sys.exit(1)

    # ── 학습 설정 출력 ────────────────────────────────────────
    print(f"\n[INFO] 학습 설정:")
    print(f"  모델:     {args.model}")
    print(f"  에폭:     {args.epochs}")
    print(f"  배치:     {args.batch} ({'자동' if args.batch == -1 else '고정'})")
    print(f"  이미지:   {args.imgsz}x{args.imgsz}")
    print(f"  디바이스: {args.device}")
    print(f"  데이터:   {data_yaml}")
    print(f"  저장:     {config.YOLO_CONFIG['project']}/{config.YOLO_CONFIG['name']}")

    # ── 학습 실행 ─────────────────────────────────────────────
    print(f"\n[INFO] 학습 시작...")
    print("  진행 상황은 실시간으로 출력됩니다.")
    print("  'results/yolo_runs/banner_detection/' 에 결과가 저장됩니다.")
    print("  Ctrl+C 로 중단 가능 (지금까지 결과는 보존됩니다)")

    results = model.train(
        # 데이터셋 설정 파일
        data=str(data_yaml),

        # 학습 에폭 수
        epochs=args.epochs,

        # 배치 크기 (-1: GPU 메모리에 맞게 자동 설정)
        batch=args.batch,

        # 입력 이미지 크기
        imgsz=args.imgsz,

        # 학습률 (lr0: 초기, lrf: 최종 = lr0 * lrf)
        lr0=config.YOLO_CONFIG["lr0"],
        lrf=config.YOLO_CONFIG["lrf"],

        # GPU 장치
        device=args.device,

        # DataLoader 워커 수 (CPU 코어 수에 맞게 조정)
        workers=config.YOLO_CONFIG["workers"],

        # Early stopping patience (에폭)
        patience=config.YOLO_CONFIG["patience"],

        # 결과 저장 디렉토리
        project=config.YOLO_CONFIG["project"],
        name=args.name,

        # 이전 학습 이어서 하기
        resume=args.resume,

        # 사전학습 가중치 사용
        pretrained=config.YOLO_CONFIG["pretrained"],

        # 최고 성능 모델 저장
        save=True,

        # 검증 주기 (매 에폭마다 검증)
        val=True,

        # AMP (Automatic Mixed Precision): FP16으로 학습 속도/메모리 개선
        amp=True,

        # 학습 시각화 결과 저장
        plots=True,
    )

    # ── 최적 모델 복사 ────────────────────────────────────────
    best_model_src = Path(config.YOLO_CONFIG["project"]) / \
                     args.name / "weights" / "best.pt"

    if best_model_src.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_src, config.YOLO_BEST_MODEL)
        print(f"\n[DONE] 최적 모델 저장: {config.YOLO_BEST_MODEL}")
    else:
        print(f"\n[WARN] best.pt를 찾을 수 없습니다: {best_model_src}")

    # ── 학습 결과 요약 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("학습 완료 - 결과 요약")
    print("=" * 60)

    # Ultralytics results 객체에서 최종 메트릭 추출
    try:
        metrics = results.results_dict
        print(f"  mAP50 (box):  {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50 (mask): {metrics.get('metrics/mAP50(M)', 0):.4f}")
        print(f"  mAP50-95:     {metrics.get('metrics/mAP50-95(M)', 0):.4f}")
    except Exception:
        print("  (결과 메트릭 파싱 실패 - 로그 파일 확인)")

    print(f"\n다음 단계: python 03_evaluate_yolo.py")

    return results


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
