"""
pipeline.py - 04_pipeline.py의 임포트 진입점
=============================================
Python 모듈 이름은 숫자로 시작할 수 없으므로
04_pipeline.py를 직접 import하기 위한 래퍼 파일입니다.

사용 예시:
    from pipeline import BannerPipeline, BannerDetector
    from pipeline import HomographyCorrector, BannerClassifier
"""

import importlib.util
import sys
from pathlib import Path

# 04_pipeline.py를 동적으로 로드
_spec = importlib.util.spec_from_file_location(
    "pipeline_impl",
    Path(__file__).parent / "04_pipeline.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_impl"] = _mod
_spec.loader.exec_module(_mod)

# 공개 심볼 re-export
BannerDetector      = _mod.BannerDetector
HomographyCorrector = _mod.HomographyCorrector
BannerClassifier    = _mod.BannerClassifier
BannerPipeline      = _mod.BannerPipeline

__all__ = [
    "BannerDetector",
    "HomographyCorrector",
    "BannerClassifier",
    "BannerPipeline",
]
