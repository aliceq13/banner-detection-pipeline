"""
config.py - 프로젝트 전역 설정 파일
=====================================
모든 스크립트에서 공통으로 사용하는 경로, 하이퍼파라미터 등을 한 곳에서 관리합니다.
설정을 변경할 때는 이 파일만 수정하면 됩니다.
"""

import os
from pathlib import Path

# ============================================================
# 기본 경로 설정
# ============================================================

# Docker 컨테이너 내부 vs 호스트 환경 자동 감지
# - Docker 내부: /workspace/project가 작업 디렉토리
# - 호스트 직접 실행: 이 파일이 있는 디렉토리
_IS_DOCKER = os.path.exists("/workspace/project")

if _IS_DOCKER:
    # Docker 컨테이너 내부 경로
    BASE_DIR         = Path("/workspace/project")
    DATA_ROOT        = Path("/workspace/illegal_banner")
    RESULTS_DIR      = Path("/workspace/results")
    HF_CACHE_DIR     = Path("/workspace/hf_cache")
else:
    # 호스트 직접 실행 경로
    BASE_DIR         = Path(__file__).parent.resolve()
    DATA_ROOT        = Path("/data/illegal_banner")
    RESULTS_DIR      = BASE_DIR / "results"
    HF_CACHE_DIR     = BASE_DIR / ".hf_cache"

# 데이터 경로
DATA_DIR         = BASE_DIR / "data"
DATASET_YAML     = DATA_DIR / "dataset.yaml"

# ── 기존 현수막 데이터 (YOLO segmentation 포맷) ─────────────
YOLO_DATA_DIR    = DATA_ROOT / "data"
YOLO_IMAGES      = YOLO_DATA_DIR / "images"
YOLO_LABELS      = YOLO_DATA_DIR / "labels"

# ── 새 현수막 데이터 (COCO JSON 포맷, 종류 레이블 포함) ─────
COCO_BANNER_DIR  = DATA_ROOT / "temp_illegal_banner" / "illegal_banner"
COCO_BANNER_IMG  = COCO_BANNER_DIR / "image" / "image"
COCO_BANNER_LBL  = COCO_BANNER_DIR / "label"

# ── COCO negative 데이터 저장 경로 ───────────────────────────
COCO_NEG_DIR     = DATA_DIR / "coco_negative"
COCO_NEG_IMAGES  = COCO_NEG_DIR / "images"
COCO_NEG_LABELS  = COCO_NEG_DIR / "labels"  # 빈 파일들 (negative)

# ── 병합된 학습 데이터 경로 ───────────────────────────────────
MERGED_DIR       = DATA_DIR / "merged"
MERGED_TRAIN_IMG = MERGED_DIR / "images" / "train"
MERGED_TRAIN_LBL = MERGED_DIR / "labels" / "train"
MERGED_VAL_IMG   = MERGED_DIR / "images" / "val"
MERGED_VAL_LBL   = MERGED_DIR / "labels" / "val"
MERGED_TEST_IMG  = MERGED_DIR / "images" / "test"
MERGED_TEST_LBL  = MERGED_DIR / "labels" / "test"

# ── VLM 분류용 데이터 경로 ───────────────────────────────────
VLM_DATA_DIR     = DATA_DIR / "vlm_classification"
VLM_TRAIN_DIR    = VLM_DATA_DIR / "train"
VLM_VAL_DIR      = VLM_DATA_DIR / "val"

# ── 모델 저장 경로 ───────────────────────────────────────────
MODELS_DIR       = RESULTS_DIR / "models"
YOLO_BEST_MODEL  = MODELS_DIR / "yolo26_banner_best.pt"

# ============================================================
# COCO Dataset 다운로드 설정
# ============================================================

# COCO 2017 val 세트 (5000장, 약 1GB) - negative sample 용
COCO_VAL_URL   = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# negative sample 최대 개수 (너무 많으면 학습이 느려짐)
COCO_NEG_MAX_SAMPLES = 2000

# ============================================================
# YOLO 학습 하이퍼파라미터
# ============================================================

YOLO_CONFIG = {
    # 모델 선택:
    # - yolo26n-seg: nano (가장 빠름, 정확도 낮음)
    # - yolo26s-seg: small (균형)
    # - yolo26m-seg: medium (더 정확, 더 느림)
    "model": "yolo26m-seg.pt",

    # 학습 에폭 수 (기본 100, 빠른 테스트시 20)
    "epochs": 100,

    # 배치 크기 (A5000 24GB 기준 -1: auto)
    "batch": -1,

    # 입력 이미지 크기 (pixels)
    "imgsz": 640,

    # 학습률
    "lr0": 0.01,
    "lrf": 0.01,

    # 데이터 로더 워커 수
    "workers": 8,

    # 조기 종료 (patience 에폭 동안 개선 없으면 중단)
    "patience": 50,

    # GPU 장치 (0: 첫 번째 GPU)
    "device": 0,

    # 결과 저장 디렉토리
    "project": str(RESULTS_DIR / "yolo_runs"),
    "name": "banner_detection",

    # 사전학습 가중치 사용 (True: ImageNet 사전학습 활용)
    "pretrained": True,
}

# ============================================================
# 클래스 정의
# ============================================================

# YOLO 세그멘테이션 클래스 (기존 데이터 기준)
YOLO_CLASSES = {
    0: "banner",   # 현수막 (segmentation 대상)
    1: "text",     # 현수막 내 텍스트 (부가 정보)
}

# VLM 분류 클래스 (현수막 종류)
# COCO JSON (temp_illegal_banner/label/*.json) 의 categories 를 실측한 결과
# 정당/민간/공공 3종만 존재. "불법" 은 이미지만으로 판별 불가한 메타데이터
# (설치 허가 여부)라 VLM 분류 대상에서 제외.
BANNER_CLASSES = {
    0: "정당 현수막",   # 정치 관련 현수막
    1: "민간 현수막",   # 개인/업체 관련 현수막
    2: "공공 현수막",   # 공공기관/지자체 현수막
}

# COCO JSON에서 사용된 카테고리 이름 → 통합 클래스 ID 매핑
COCO_CAT_TO_CLASS = {
    "정당 현수막": 0,
    "민간 현수막": 1,
    "공공 현수막": 2,
}

# ============================================================
# VLM (Qwen3.5-9B 등) 설정
# ============================================================

VLM_CONFIG = {
    # 모델 ID (HuggingFace Hub)
    # Qwen3.5-9B: 9B 파라미터, 이미지·텍스트·비디오 멀티모달 (Image-Text-to-Text).
    # 하이브리드 아키텍처 (Gated DeltaNet + Gated Attention), context 262K 토큰.
    # BF16 ~18GB — A5000 24GB에 YOLO(~500MB)와 동시 상주 가능하므로 양자화 불필요.
    "model_id": "Qwen/Qwen3.5-9B",

    # 양자화 설정 (메모리 절약)
    # - None  : BF16/FP16 원본 (정확도 최대, 24GB 안에 9B가 들어오면 선호)
    # - "8bit": INT8 양자화 (~10GB, 정확도 약간 손실)
    # - "4bit": NF4 양자화 (~6GB, 정확도 더 손실)
    "quantization": None,

    # 추론 디바이스
    "device": "cuda",

    # 최대 생성 토큰 수.
    # Qwen3.5-9B 의 thinking mode 가 켜지면 본답변 전에 <think>...</think> 블록이
    # 수백 토큰 나오므로 충분히 크게 둬야 한다. 256 이면 think 체인이 버짓을 다
    # 먹고 "카테고리:" 라인이 잘리는 사고 발생. 1024 로 상향.
    "max_new_tokens": 1024,

    # 온도 (낮을수록 결정적, 0.1~0.3 권장)
    "temperature": 0.1,

    # 모델 캐시 경로
    "cache_dir": str(HF_CACHE_DIR),
}

# ============================================================
# H-Matrix (호모그래피) 설정
# ============================================================

HMATRIX_CONFIG = {
    # 보정 이미지의 긴 변 상한 (픽셀).
    # 실제 출력 W/H는 세그 폴리곤에서 뽑은 4 극점 간 거리로 계산되며,
    # 계산 결과가 max_size보다 크면 비율 유지한 채 균등 축소한다.
    "max_size": 1024,
}

# ============================================================
# 경로 생성 함수
# ============================================================

def ensure_dirs():
    """필요한 디렉토리들을 모두 생성합니다."""
    dirs_to_create = [
        DATA_DIR,
        COCO_NEG_DIR, COCO_NEG_IMAGES, COCO_NEG_LABELS,
        MERGED_DIR,
        MERGED_TRAIN_IMG, MERGED_TRAIN_LBL,
        MERGED_VAL_IMG, MERGED_VAL_LBL,
        MERGED_TEST_IMG, MERGED_TEST_LBL,
        VLM_DATA_DIR, VLM_TRAIN_DIR, VLM_VAL_DIR,
        RESULTS_DIR, MODELS_DIR,
        HF_CACHE_DIR,
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[config] 필요한 디렉토리 생성 완료: {BASE_DIR}")


if __name__ == "__main__":
    # 설정 확인용 출력
    print("=" * 60)
    print("프로젝트 설정 확인")
    print("=" * 60)
    print(f"실행 환경: {'Docker 컨테이너' if _IS_DOCKER else '호스트 직접 실행'}")
    print(f"BASE_DIR:  {BASE_DIR}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"RESULTS:   {RESULTS_DIR}")
    print()
    print("[YOLO 데이터]")
    print(f"  이미지: {YOLO_IMAGES} (존재: {YOLO_IMAGES.exists()})")
    print(f"  레이블: {YOLO_LABELS} (존재: {YOLO_LABELS.exists()})")
    print()
    print("[COCO 현수막 데이터]")
    print(f"  이미지: {COCO_BANNER_IMG} (존재: {COCO_BANNER_IMG.exists()})")
    print(f"  레이블: {COCO_BANNER_LBL} (존재: {COCO_BANNER_LBL.exists()})")
    print()
    print("[모델 설정]")
    print(f"  YOLO:  {YOLO_CONFIG['model']}")
    print(f"  VLM:   {VLM_CONFIG['model_id']}")
    ensure_dirs()
