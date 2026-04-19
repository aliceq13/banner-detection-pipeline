# ============================================================
# Dockerfile - 현수막 탐지 및 분류 프로젝트
# ============================================================
# 베이스 이미지: PyTorch + CUDA 12.1 공식 이미지
# NVIDIA RTX A5000 (24GB VRAM) 기준으로 설정
# ============================================================

# runtime 버전 사용: devel 대비 ~3GB 작음, 학습/추론에 충분
# 이미 로컬에 있어서 추가 다운로드 없음
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ── 메타데이터 ──────────────────────────────────────────────
LABEL maintainer="banner-detection-project"
LABEL description="현수막 탐지(YOLO26) + H-matrix 보정 + VLM 분류(Qwen3.5-9B) 파이프라인"

# ── 환경 변수 설정 ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
# Python 출력 버퍼링 비활성화 (로그 실시간 확인)
ENV PYTHONUNBUFFERED=1
# UTF-8 인코딩 (한글 처리)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
# HuggingFace 캐시 디렉토리 (모델 저장 위치)
ENV HF_HOME=/workspace/hf_cache
# Ultralytics 설정 디렉토리
ENV YOLO_CONFIG_DIR=/workspace/yolo_config

# ── 시스템 패키지 설치 ──────────────────────────────────────
# - libgl1-mesa-glx: OpenCV GUI 지원
# - libglib2.0-0: GLib 라이브러리 (OpenCV 의존성)
# - wget, curl: 파일 다운로드
# - git: 소스코드 클론
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    git \
    zip \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── 작업 디렉토리 설정 ─────────────────────────────────────
WORKDIR /workspace

# ── Python 패키지 설치 ─────────────────────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── 프로젝트 코드 복사 ─────────────────────────────────────
# (docker-compose에서 볼륨 마운트로 대체 가능)
COPY . /workspace/project/

# ── 포트 노출 (Jupyter Notebook 용) ───────────────────────
EXPOSE 8888

# ── 기본 실행 커맨드 ──────────────────────────────────────
# 기본으로 bash 실행 (docker-compose에서 override)
CMD ["/bin/bash"]
