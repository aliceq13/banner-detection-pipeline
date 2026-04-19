"""
pipeline_utils.py - 파이프라인 보조 유틸리티
=============================================
04_pipeline.py와 05_demo.py에서 공통으로 사용하는 함수들을 모아둡니다.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config


def apply_hmatrix_to_crop(
    crop: np.ndarray,
    max_size: int = config.HMATRIX_CONFIG["max_size"],
) -> np.ndarray:
    """
    이미 crop된 현수막 이미지를 max_size 한도 내에서 종횡비 보존 리사이즈한다.

    crop은 이미 직사각형이라 원근 보정할 4 꼭짓점 정보가 없다. 세그 다각형이
    있는 경로(HomographyCorrector.correct)와 달리, 여기서는 실제 H-matrix
    연산이 필요 없고 크기만 max_size에 맞춘다.

    Args:
        crop: BGR 이미지 (이미 crop된 현수막)
        max_size: 긴 변 상한 (픽셀)

    Returns:
        리사이즈된 BGR 이미지
    """
    h, w = crop.shape[:2]
    if max(h, w) <= max_size:
        return crop
    scale = max_size / float(max(h, w))
    return cv2.resize(
        crop,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
