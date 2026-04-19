"""
04_pipeline.py - 현수막 탐지 및 분류 전체 파이프라인
=====================================================
3-단계 파이프라인:
  1. [YOLO26-seg]   현수막 영역 세그멘테이션 (polygon + mask)
  2. [H-Matrix]     polygon에서 4 극점 추출 → 종횡비 보존 원근 보정
  3. [VLM]          Qwen/Qwen3.5-9B 등 멀티모달 LLM으로 3-class 분류
                    (정당/민간/공공 현수막 — 데이터 라벨과 정합)

세부:
  YOLO26-seg (BannerDetector):
    - ultralytics.YOLO 래퍼. result.masks.xy[0]로 폴리곤(픽셀 좌표),
      mask.data[0]로 boolean 마스크, box.xyxy로 bbox를 함께 반환.

  H-Matrix (HomographyCorrector):
    - polygon N개 점 전부에서 TL=argmin(x+y), TR=argmax(x-y),
      BR=argmax(x+y), BL=argmin(x-y) 4 극점을 직접 선택
      (convexHull/approxPolyDP/회전 로직 없음).
    - 출력 W,H = 4 극점 사이 실측 변 길이 (max_size 상한 1024 px,
      초과 시 비율 유지 축소). 4점 정확해인 cv2.getPerspectiveTransform
      + cv2.warpPerspective로 변환.

  VLM (BannerClassifier):
    - 모델 ID는 config.VLM_CONFIG["model_id"] (기본 Qwen/Qwen3.5-9B,
      멀티모달 image-text-to-text, BF16 원본 ~18GB).
    - 기본은 양자화 없이 BF16 로드. 24GB A5000 에 YOLO(~500MB)와 동시 상주.
      config.VLM_CONFIG["quantization"] 을 "4bit"/"8bit" 로 바꾸면 bitsandbytes
      양자화 경로로 폴백 가능.
    - 입력 이미지 긴 변을 MAX_VLM_IMAGE_SIDE(=896px) 로 축소. 큰 이미지에서
      "!!!!..." 같은 gibberish 가 나오는 Qwen3.5-9B 공식 디스커션 #12 의
      권고를 반영.
    - processor.apply_chat_template(messages, enable_thinking=True,
      add_generation_prompt=True, tokenize=True, return_dict=True)로 이미지
      토큰 주입 + thinking mode ON. Qwen3 family 는 enable_thinking 을
      직접 kwarg 로 받는다 (chat_template_kwargs 딕셔너리는 AutoProcessor
      단계에서 무시됨). Thinking 체인은 <think>...</think> 블록이므로
      생성 후 _strip_think_block 으로 떼고 본답변만 파싱.
      max_new_tokens 는 config 에서 1024 로 상향해 think 예산 확보.
    - 프롬프트는 3단 구조: (1) 현수막 문구 추출 → (2) 근거 → (3) 카테고리.
      모델이 이미지를 실제로 "읽는지" 문구 라인으로 검증 가능.
    - 파싱은 "카테고리:" 라인 → 답변 말미 5줄 → 전체 텍스트 → 키워드 순
      폴백으로 다단 매칭. 단일 글자("당") 같은 과매칭 키워드는 제거.
    - 모든 단계 try/except 로 감싸 실패해도 유효 dict 반환 (배치 평가가
      한 샘플 예외로 멈추지 않음).

실행:
  python 04_pipeline.py --image <path>
  python 04_pipeline.py --image <path> --visualize
  python 04_pipeline.py --batch <dir>
  python 04_pipeline.py --image <path> --no-vlm   # VLM 건너뜀
"""

import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config


# ============================================================
# Step 1: YOLO26 현수막 세그멘테이션
# ============================================================

class BannerDetector:
    """
    YOLO26 기반 현수막 탐지기.

    세그멘테이션 마스크(픽셀 단위 경계)를 출력합니다.
    일반 객체 탐지(bounding box)와 달리 현수막의 정확한 모양을 파악할 수 있어
    이후 H-matrix 계산에 유리합니다.
    """

    def __init__(
        self,
        model_path: str = str(config.YOLO_BEST_MODEL),
        conf: float = 0.25,
        iou: float = 0.6,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: 학습된 YOLO 모델 경로
            conf: 신뢰도 임계값 (0~1, 높을수록 확실한 것만 탐지)
            iou: NMS IoU 임계값 (겹치는 탐지 제거 기준)
            device: 추론 디바이스 ("cuda" or "cpu")
        """
        from ultralytics import YOLO

        self.conf   = conf
        self.iou    = iou
        self.device = device

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO 모델 없음: {model_path}\n"
                "02_train_yolo.py로 먼저 모델을 학습하세요."
            )

        print(f"[BannerDetector] 모델 로드: {model_path}")
        self.model = YOLO(str(model_path))
        print(f"[BannerDetector] 준비 완료 (conf={conf}, iou={iou})")

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        이미지에서 현수막을 탐지합니다.

        Args:
            image: BGR 이미지 (OpenCV 형식)

        Returns:
            탐지된 현수막 목록, 각 항목:
            {
                'class_id':  0 (banner) or 1 (text),
                'class_name': 'banner' or 'text',
                'confidence': 0.0~1.0,
                'bbox':       [x1, y1, x2, y2] (픽셀 절대좌표),
                'mask':       np.ndarray (H x W, bool),
                'polygon':    np.ndarray (N x 2, 픽셀 좌표),
            }
        """
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.masks is None:
                continue  # 세그멘테이션 마스크 없으면 건너뜀

            # 각 탐지 객체 처리
            for box, mask in zip(result.boxes, result.masks):
                cls_id   = int(box.cls[0])
                cls_name = config.YOLO_CLASSES.get(cls_id, f"class_{cls_id}")

                # 세그멘테이션 다각형 좌표 (normalized → pixel)
                polygon = mask.xy[0]  # shape: (N, 2), 픽셀 좌표

                # bounding box (xyxy 형식)
                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                # boolean 마스크 (H x W)
                bool_mask = mask.data[0].cpu().numpy().astype(bool)

                detections.append({
                    'class_id':   cls_id,
                    'class_name': cls_name,
                    'confidence': float(box.conf[0]),
                    'bbox':       bbox,
                    'mask':       bool_mask,
                    'polygon':    polygon,
                })

        return detections


# ============================================================
# Step 2: H-Matrix 원근 보정
# ============================================================

class HomographyCorrector:
    """
    세그 폴리곤에서 4 극점을 직접 골라 H-matrix 원근 보정을 수행한다.

    흐름:
    1. polygon N개 점 전체에서 TL=argmin(x+y), TR=argmax(x−y),
       BR=argmax(x+y), BL=argmin(x−y) 4점 선택.
       (convexHull / approxPolyDP 같은 단순화나 임의 회전 없음.)
    2. 상/하 변 중 긴 쪽을 W, 좌/우 변 중 긴 쪽을 H로 취해 목표 직사각형
       크기를 결정 — 이 4점이 만드는 쿼드의 종횡비를 그대로 따라간다.
    3. max_size 초과 시 비율 유지하며 균등 축소.
    4. cv2.getPerspectiveTransform(4점 정확해) + cv2.warpPerspective 로 변환.
    """

    def __init__(
        self,
        max_size: int = config.HMATRIX_CONFIG["max_size"],
    ):
        """
        Args:
            max_size: 보정 이미지의 긴 변 상한 (픽셀). W/H가 이보다 크면
                      종횡비를 유지하며 축소한다.
        """
        self.max_size = max_size

    def correct(
        self,
        image:   np.ndarray,
        polygon: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        다각형 꼭짓점을 이용해 이미지를 원근 보정합니다.

        Args:
            image:   원본 BGR 이미지
            polygon: 세그멘테이션 다각형 (N x 2 배열, 픽셀 좌표)

        Returns:
            (보정된 이미지, H 행렬) 또는 (None, None) (실패시)
        """
        if len(polygon) < 4:
            return None, None

        ordered = self._pick_4_extremes(polygon)
        if ordered is None:
            return None, None

        TL, TR, BR, BL = ordered
        W = int(round(max(np.linalg.norm(TR - TL), np.linalg.norm(BR - BL))))
        H_out = int(round(max(np.linalg.norm(BL - TL), np.linalg.norm(BR - TR))))
        if W < 2 or H_out < 2:
            return None, None

        if max(W, H_out) > self.max_size:
            s = self.max_size / max(W, H_out)
            W, H_out = int(round(W * s)), int(round(H_out * s))

        dst_pts = np.float32([
            [0,     0      ],
            [W - 1, 0      ],
            [W - 1, H_out - 1],
            [0,     H_out - 1],
        ])

        src_pts = ordered.astype(np.float32)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        corrected = cv2.warpPerspective(
            image, H, (W, H_out),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return corrected, H

    def _pick_4_extremes(self, polygon: np.ndarray) -> Optional[np.ndarray]:
        """세그 폴리곤 전체 점 중에서 4 극점을 직접 고른다.

          TL = argmin(x + y)   (좌·상단에 가장 가까운 점)
          TR = argmax(x - y)   (우·상단)
          BR = argmax(x + y)   (우·하단)
          BL = argmin(x - y)   (좌·하단)

        convexHull + approxPolyDP 로 단순화하지 않고 polygon N개 점 전부를
        후보로 보기 때문에 불규칙한 마스크에서도 극점이 흔들리지 않는다.
        반환 순서는 [TL, TR, BR, BL].
        """
        pts = polygon.reshape(-1, 2).astype(np.float32)
        if len(pts) < 4:
            return None
        s = pts[:, 0] + pts[:, 1]   # x + y
        d = pts[:, 0] - pts[:, 1]   # x - y
        return np.stack([
            pts[np.argmin(s)],  # TL
            pts[np.argmax(d)],  # TR
            pts[np.argmax(s)],  # BR
            pts[np.argmin(d)],  # BL
        ], axis=0)


# ============================================================
# Step 3: VLM 분류 (Qwen3.5-9B 등)
# ============================================================

class BannerClassifier:
    """
    멀티모달 VLM으로 현수막 종류를 분류한다.

    - 모델 ID는 config.VLM_CONFIG["model_id"] (기본 Qwen/Qwen3.5-9B,
      9B 파라미터 image-text-to-text 멀티모달, BF16 ~18GB).
    - 입력 이미지는 긴 변 MAX_VLM_IMAGE_SIDE(=896px) 로 축소 후 전달
      (큰 이미지에서 Qwen3.5-9B 가 gibberish 를 내뱉는 알려진 문제 회피,
       공식 디스커션 #12).
    - 프롬프트는 processor.apply_chat_template 로 이미지 토큰과 함께 주입
      (그래야 'image tokens: 0, features: N' 미스매치가 나지 않는다).
      Qwen3 family 규약에 따라 `enable_thinking=True` 를 **직접 kwarg** 로
      전달해 thinking mode ON. (과거 `chat_template_kwargs` 딕셔너리로
      전달하면 AutoProcessor 가 "not a valid argument ... will be ignored"
      로 무시했음.) 미지원 버전은 TypeError 로 감지해 기본값으로 폴백.
    - Thinking chain 은 <think>...</think> 블록으로 길게 나오므로
      max_new_tokens 는 config.py 에서 1024 로 상향. 생성 결과에서
      _strip_think_block 으로 블록을 떼고 본답변만 파싱한다.
    - 프롬프트 3단 구조: (1) 현수막 문구 읽어서 옮겨 쓰기, (2) 근거 설명,
      (3) 고정 포맷 "카테고리: ..." 라인. (1) 번으로 이미지 OCR 여부 확인.
    - 예외: 전처리/템플릿/generate/디코딩 각 단계를 try/except 로 감싸
      실패 시에도 정상 dict(`category="민간 현수막", confidence=0`) 반환 —
      배치 평가(`05_demo.py`) 가 한 샘플 실패로 전체 중단되지 않도록.
    - 기본 로드는 BF16 원본 (양자화 없음). 24GB A5000 에 여유 있게 들어오고
      정확도 손실이 없어 양자화보다 선호. config.VLM_CONFIG["quantization"] 을
      "4bit"/"8bit" 로 바꾸면 bitsandbytes 양자화 경로로 폴백 가능.
    """

    # 분류 카테고리 (config.BANNER_CLASSES 에서 자동 도출) 와 파싱용 키워드 사전.
    # "불법" 은 이미지만으로 판별 불가한 메타데이터이므로 분류 대상에서 제외.
    # 단일 글자 키워드("당")는 "당신/당첨/당일" 같은 무관한 단어까지 매칭해
    # 오탐을 유도했으므로 의미 단위(2글자+) 키워드만 남긴다.
    CATEGORY_KEYWORDS = {
        "정당 현수막":  ["정당", "정치", "선거", "후보", "국회", "의원"],
        "민간 현수막":  ["민간", "개인", "업체", "가게", "광고", "홍보", "상점", "매장"],
        "공공 현수막":  ["공공", "기관", "시청", "구청", "행정", "복지", "지자체", "정부"],
    }

    # Qwen3.5-9B는 입력 이미지가 너무 크면 이미지 토큰이 폭증하면서 생성이
    # 무너진다(공식 디스커션 #12: "gibberish response" 리포트의 합의된 원인).
    # 긴 변이 이 값보다 크면 종횡비를 유지한 채 축소한 뒤 모델에 전달한다.
    MAX_VLM_IMAGE_SIDE = 896

    def __init__(
        self,
        model_id: str = config.VLM_CONFIG["model_id"],
        quantization: Optional[str] = config.VLM_CONFIG["quantization"],
        device: str = config.VLM_CONFIG["device"],
        max_new_tokens: int = config.VLM_CONFIG["max_new_tokens"],
        temperature: float = config.VLM_CONFIG["temperature"],
        cache_dir: str = config.VLM_CONFIG["cache_dir"],
    ):
        """
        Args:
            model_id: HuggingFace 모델 ID
            quantization: 양자화 ("4bit", "8bit", None)
            device: 추론 디바이스
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 출력 다양성 (낮을수록 결정적)
            cache_dir: 모델 캐시 디렉토리
        """
        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.device         = device

        print(f"\n[BannerClassifier] loading VLM: {model_id}")
        print(f"  quant: {quantization}, device: {device}")

        self.processor, self.model = self._load_model(
            model_id, quantization, device, cache_dir
        )
        print(f"[BannerClassifier] ready")

    def _load_model(self, model_id, quantization, device, cache_dir):
        """HuggingFace에서 VLM을 로드한다.

        - 로더: AutoModelForImageTextToText 우선 (Qwen3.5-9B 등 image-text-to-text
          분류 모델), 미래형 AutoModelForMultimodalLM 으로 fallback.
        - dtype : BF16 원본 (Qwen3.5-9B 가중치가 BF16). 양자화 시에도 compute dtype
          은 BF16 유지.
        - 양자화(bitsandbytes):
            None  : BF16 원본 (기본, ~18GB)
            "8bit": load_in_8bit (~10GB)
            "4bit": NF4 + double-quant (~6GB)
        """
        import torch
        from transformers import AutoProcessor

        # Qwen3.5-9B 은 HF 에서 image-text-to-text 로 분류됨 →
        # AutoModelForImageTextToText 우선. 미래형 통합 로더
        # AutoModelForMultimodalLM 이 존재하면 그걸 사용해 재포워드 호환.
        try:
            from transformers import AutoModelForImageTextToText
            ModelCls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForMultimodalLM
            ModelCls = AutoModelForMultimodalLM

        # 양자화 설정 — Qwen3.5-9B는 BF16 네이티브이므로 compute dtype도 BF16
        quant_config = None
        torch_dtype  = torch.bfloat16

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # 이중 양자화로 추가 압축
                bnb_4bit_quant_type="nf4",       # NF4: 정규분포에 최적화된 양자화
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # 프로세서 로드 (이미지 전처리 + 토크나이저)
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # 모델 로드
        # device_map={"": 0} : accelerate의 "auto" 가 보수적으로 CPU/disk offload
        # 를 결정해 bitsandbytes 양자화 경로에서 "Some modules are dispatched on the
        # CPU" 에러를 내는 문제를 우회. Qwen3.5-9B BF16 ~18GB 는 A5000 24GB 에
        # 여유 있게 들어오지만, 양자화 경로에서도 안전하도록 단일 GPU 고정 유지.
        load_kwargs = {
            "cache_dir": cache_dir,
            "device_map": {"": 0},
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        model = ModelCls.from_pretrained(
            model_id,
            **load_kwargs,
        )
        model.eval()  # 추론 모드 (드롭아웃 비활성화)

        return processor, model

    def classify(self, image: np.ndarray, debug: bool = False) -> dict:
        """
        현수막 이미지를 VLM으로 분류합니다.

        처리 흐름:
          1. BGR→RGB→PIL 변환 후, 긴 변을 MAX_VLM_IMAGE_SIDE 로 축소
             (Qwen3.5-9B 가 큰 이미지에서 "!!!!..." 같은 gibberish 를
             내뱉는 알려진 문제를 회피. 공식 디스커션 #12).
          2. 이미지 콘텐츠 + 텍스트 요청을 하나의 user message 로 묶어
             processor.apply_chat_template 로 이미지 토큰을 명시 주입하면서
             **thinking mode 를 켠다(`enable_thinking=True`)**.
             - Qwen3 family 의 chat template 은 `enable_thinking` 을 *직접
               kwarg* 로 받는다 (블로그 "The 4 things Qwen-3's chat template
               teaches us" 참조). 과거에 시도한 `chat_template_kwargs=
               {"enable_thinking": False}` 는 AutoProcessor 에서 "not a
               valid argument ... will be ignored" 로 무시되던 버그 경로.
             - 지원하지 않는 구버전 transformers 에서는 TypeError 로 폴백.
          3. Thinking 체인은 <think>...</think> 블록으로 길게 나오므로
             max_new_tokens 는 config.py 에서 1024 로 확장. 생성 후
             _strip_think_block 으로 blok 을 떼고 본답변만 파싱.
          4. generate 단계에서 repetition_penalty=1.1 로 "!!!!..." 억제.
          5. 디코드 → think blok 제거 → _parse_category 로 3-class 분류 추출.

        Args:
            image: BGR 이미지 (OpenCV, H-matrix 보정 후)
            debug: True 면 입력 토큰 수 / feature 수 / think 트레이스 길이 등
                   진단 로그를 콘솔에 출력

        Returns:
            {
                'category':    '정당 현수막' 등,
                'confidence':  0.0~1.0 (키워드 기반 신뢰도),
                'raw_output':  모델 원본 출력 텍스트 (think blok 포함),
                'answer':      think blok 을 제거한 본답변,
                'think':       <think>...</think> 내부의 reasoning trace,
                'reason':      분류 근거 ("근거:" 라인),
                'extracted_text': 모델이 추출한 현수막 문구 ("문구:" 라인),
            }
        """
        import torch

        # ── 이미지 전처리 ─────────────────────────────────────
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            w, h = pil_image.size
            long_side = max(w, h)
            if long_side > self.MAX_VLM_IMAGE_SIDE:
                scale = self.MAX_VLM_IMAGE_SIDE / long_side
                new_size = (max(1, int(round(w * scale))),
                            max(1, int(round(h * scale))))
                pil_image = pil_image.resize(new_size, Image.BICUBIC)
        except Exception as e:
            return self._error_result(f"이미지 전처리 실패: {e}")

        # ── 메시지 구성 + chat template ───────────────────────
        prompt_text = self._build_prompt()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text": prompt_text},
                ],
            }
        ]

        template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Qwen3 family 규약: enable_thinking 은 직접 kwarg.
        # 이 kwarg 를 모르는 processor 버전이면 TypeError → 폴백.
        try:
            inputs = self.processor.apply_chat_template(
                messages, enable_thinking=True, **template_kwargs,
            )
        except TypeError as e:
            if debug:
                print(f"    [VLM debug] enable_thinking kwarg 미지원 → 기본값 사용 ({e})")
            try:
                inputs = self.processor.apply_chat_template(
                    messages, **template_kwargs,
                )
            except Exception as e2:
                return self._error_result(f"apply_chat_template 실패: {e2}")
        except Exception as e:
            return self._error_result(f"apply_chat_template 실패: {e}")

        try:
            inputs = inputs.to(self.model.device)
        except Exception as e:
            return self._error_result(f"inputs.to(device) 실패: {e}")

        if debug:
            n_tokens = inputs["input_ids"].shape[1]
            pixel_shape = tuple(inputs["pixel_values"].shape) \
                if "pixel_values" in inputs else None
            print(f"    [VLM debug] thinking=ON, input_tokens={n_tokens}, "
                  f"image_size={pil_image.size}, pixel_values={pixel_shape}, "
                  f"max_new_tokens={self.max_new_tokens}")

        # ── 생성 ──────────────────────────────────────────────
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    top_p=0.9,
                    repetition_penalty=1.1,      # "!!!!..." 억제
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
        except Exception as e:
            return self._error_result(f"generate 실패: {e}")

        try:
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            raw_text = self.processor.decode(generated, skip_special_tokens=True)
        except Exception as e:
            return self._error_result(f"디코딩 실패: {e}")

        # ── think blok 분리 ───────────────────────────────────
        think_text, answer_text = self._strip_think_block(raw_text)

        if debug:
            print(f"    [VLM debug] think_len={len(think_text)} chars, "
                  f"answer_len={len(answer_text)} chars")
            print(f"    [VLM debug] answer[:200]={answer_text[:200]!r}")

        category, confidence = self._parse_category(answer_text)

        return {
            'category':       category,
            'confidence':     confidence,
            'raw_output':     raw_text,
            'answer':         answer_text,
            'think':          think_text,
            'reason':         self._extract_reason(answer_text),
            'extracted_text': self._extract_banner_text(answer_text),
        }

    @staticmethod
    def _strip_think_block(text: str) -> Tuple[str, str]:
        """Qwen3 thinking mode 출력에서 <think>...</think> 블록을 떼낸다.

        Returns:
            (think_content, answer_text): think 블록이 없으면 첫 값은 "".
            </think> 태그만 있고 여는 태그가 없는 경우(템플릿에 이미 주입된
            think 헤더가 있을 때)도 처리.
        """
        import re
        if not text:
            return "", ""
        # 1) 정상 케이스: <think>...</think>
        m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if m:
            think = m.group(1).strip()
            answer = (text[:m.start()] + text[m.end():]).strip()
            return think, answer
        # 2) 닫는 태그만 있는 케이스 (템플릿이 <think> 를 미리 삽입한 경우)
        if "</think>" in text:
            idx = text.index("</think>")
            return text[:idx].strip(), text[idx + len("</think>"):].strip()
        # 3) 닫히지 않은 <think> 만 있는 케이스 (생성 예산 초과로 truncate)
        if "<think>" in text:
            idx = text.index("<think>")
            return text[idx + len("<think>"):].strip(), text[:idx].strip()
        return "", text.strip()

    def _error_result(self, msg: str) -> dict:
        """classify 내부 에러 발생 시 배치 평가가 멈추지 않도록 기본 dict 반환."""
        print(f"    [VLM ERROR] {msg}")
        return {
            'category':       "민간 현수막",
            'confidence':     0.0,
            'raw_output':     f"[ERROR] {msg}",
            'answer':         "",
            'think':          "",
            'reason':         "VLM 추론 실패",
            'extracted_text': "",
        }

    def _build_prompt(self) -> str:
        """
        현수막 분류를 위한 프롬프트를 생성한다.

        구조 — 3단계 구조적 응답 (Chain-of-Thought + 고정 출력 포맷):
          1) 먼저 현수막의 한글 문구를 그대로 옮겨 적게 해 모델이 정말로
             이미지를 '읽고' 있는지 확인할 수 있게 한다 (같은 응답이
             반복되는 'Party only' 증상의 진단 근거).
          2) 추출한 문구와 로고·색상을 근거로 어느 카테고리인지 추론.
          3) 마지막에 고정 포맷 "카테고리: …" 줄을 강제해 파서가 안정적.

        실제 학습 데이터(COCO JSON) 에 존재하는 3 카테고리(정당/민간/공공)
        만 선택지로 제시. "불법" 같은 비시각적 메타데이터는 이미지만으로
        판별 불가이므로 제외.
        """
        return (
            "당신은 한국 거리 현수막을 분류하는 전문가입니다. "
            "아래 순서대로 한국어로 답변하세요.\n\n"
            "## 1단계: 문구 추출\n"
            "현수막에 쓰인 한글 문구를 읽히는 대로 옮겨 적으세요. "
            "여러 줄이면 줄바꿈으로 구분하세요. 식별이 불가하면 \"없음\".\n\n"
            "## 2단계: 근거\n"
            "추출한 문구, 로고/색상, 디자인을 근거로 어느 카테고리인지 "
            "2문장 이내로 설명하세요.\n\n"
            "## 3단계: 분류 (아래 3개 중 하나만)\n"
            "- 정당 현수막: 정당/선거/정치인/후보자/국회의원 관련\n"
            "- 민간 현수막: 상점/기업/개인 광고·홍보·행사\n"
            "- 공공 현수막: 시청/구청/행정기관/공공 캠페인\n\n"
            "## 출력 포맷 (반드시 이 순서와 라벨 그대로)\n"
            "문구: <추출한 문구>\n"
            "근거: <한두 문장>\n"
            "카테고리: <정당 현수막 | 민간 현수막 | 공공 현수막>\n"
        )

    def _parse_category(self, text: str) -> Tuple[str, float]:
        """
        VLM 출력 텍스트에서 카테고리를 추출합니다.

        파싱 우선순위:
          1. "카테고리:" 라인에서 3 카테고리 이름을 정확히 매칭 (가장 신뢰).
          2. "카테고리:" 라인에서 키워드(정당/민간/공공) 매칭.
          3. (마지막 5줄만 본) 전체 텍스트에서 카테고리 이름 직접 매칭 —
             프롬프트 echo 로 상단이 오염돼도 실제 답변부만 본다.
          4. 그래도 못 찾으면 전역 키워드 매칭.
          5. 최종 실패시 "민간 현수막"(가장 흔한 class) 로 기본값.

        Args:
            text: VLM 원본 출력 텍스트

        Returns:
            (카테고리명, 신뢰도 0~1)
        """
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

        # 1·2순위: "카테고리:" 라인만 보기 (가장 신뢰)
        category_line = None
        for ln in lines:
            low = ln.lower()
            if ln.startswith("카테고리:") or ln.startswith("분류:") \
                    or low.startswith("category:"):
                category_line = ln
                break
        if category_line is not None:
            for category in self.CATEGORY_KEYWORDS:
                if category in category_line:
                    return category, 0.95
            for category, keywords in self.CATEGORY_KEYWORDS.items():
                if any(kw in category_line for kw in keywords):
                    return category, 0.85

        # 3순위: 답변의 마지막 5줄(고정 포맷의 결론부)에서 카테고리 이름 탐색
        tail = '\n'.join(lines[-5:])
        for category in self.CATEGORY_KEYWORDS:
            if category in tail:
                return category, 0.7

        # 4순위: 전체 텍스트에서 카테고리 이름 매칭
        for category in self.CATEGORY_KEYWORDS:
            if category in text:
                return category, 0.55

        # 5순위: 키워드 기반 글로벌 매칭 (정확도 낮음)
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return category, 0.35

        return "민간 현수막", 0.1

    def _extract_reason(self, text: str) -> str:
        """
        VLM 출력에서 분류 이유("근거:" 라인)를 추출합니다.

        프롬프트에서 "근거:" 를 요청하므로 그 라인을 우선 보고, 없으면
        구 버전 포맷인 "이유:" 로 폴백. 둘 다 없으면 본문 앞부분을 반환.
        """
        for line in text.split('\n'):
            low = line.lower().strip()
            if line.strip().startswith("근거:") or line.strip().startswith("이유:") \
                    or low.startswith("reason:"):
                reason = line.split(':', 1)[-1].strip()
                if reason:
                    return reason
        return text[:200] if text else "이유 없음"

    def _extract_banner_text(self, text: str) -> str:
        """VLM 출력에서 "문구:" 라인 (현수막에서 추출한 문구) 을 뽑는다."""
        for line in text.split('\n'):
            s = line.strip()
            if s.startswith("문구:") or s.lower().startswith("text:"):
                return s.split(':', 1)[-1].strip()
        return ""


# ============================================================
# 전체 파이프라인
# ============================================================

class BannerPipeline:
    """
    현수막 탐지 → 원근 보정 → VLM 분류 통합 파이프라인.

    사용 예시:
        pipeline = BannerPipeline()
        results = pipeline.process("banner.jpg")
        for r in results:
            print(r['category'], r['confidence'])
    """

    def __init__(
        self,
        yolo_model_path: str = str(config.YOLO_BEST_MODEL),
        load_vlm: bool = True,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.6,
    ):
        """
        Args:
            yolo_model_path: YOLO 모델 경로
            load_vlm: VLM 로드 여부 (False면 탐지만 수행)
            yolo_conf: YOLO 신뢰도 임계값
            yolo_iou: YOLO IoU 임계값
        """
        # YOLO 탐지기 초기화
        self.detector = BannerDetector(
            model_path=yolo_model_path,
            conf=yolo_conf,
            iou=yolo_iou,
        )

        # H-matrix 보정기 초기화
        self.corrector = HomographyCorrector()

        # VLM 분류기 초기화 (선택적)
        self.classifier = None
        if load_vlm:
            try:
                self.classifier = BannerClassifier()
            except Exception as e:
                print(f"[WARNING] VLM 로드 실패: {e}")
                print("  탐지 + 보정만 수행합니다.")

    def unload_detector(self):
        """YOLO 모델을 VRAM에서 내린다.

        Qwen3.5-9B BF16(~18GB) 은 A5000 24GB 에 YOLO 와 같이 들어오므로 대부분
        해제가 필수는 아니지만, 05_demo.py 처럼 "모든 감지를 먼저 수행한 뒤
        VLM 로드"하는 흐름에서 여유 VRAM 을 확보해 추론 안정성을 높인다. 양자화
        없이 18GB 를 올릴 때 가장자리 여유가 중요하기 때문. 해제 후 detect()
        호출은 불가.
        """
        import torch, gc
        if self.detector is not None:
            try:
                del self.detector.model
            except AttributeError:
                pass
            self.detector = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_classifier(self):
        """VLM을 지연 로드한다 (이미 로드돼 있으면 no-op).

        detector를 먼저 해제한 뒤 호출해야 OOM/offload 에러를 피할 수 있다.
        """
        if self.classifier is not None:
            return
        try:
            self.classifier = BannerClassifier()
        except Exception as e:
            print(f"[WARNING] VLM 로드 실패: {e}")
            print("  분류 단계는 건너뜁니다.")

    def process(
        self,
        image_path: str,
        visualize: bool = False,
        save_dir: Optional[str] = None,
    ) -> List[dict]:
        """
        이미지에서 현수막을 탐지하고 분류합니다.

        Args:
            image_path: 입력 이미지 경로
            visualize: 중간 결과 시각화 여부
            save_dir: 결과 저장 디렉토리 (None이면 저장 안 함)

        Returns:
            탐지된 현수막 목록, 각 항목:
            {
                'detection':  YOLO 탐지 결과,
                'corrected':  H-matrix 보정 이미지 (np.ndarray),
                'H_matrix':   3x3 호모그래피 행렬,
                'classification': VLM 분류 결과,
            }
        """
        # ── 이미지 로드 ───────────────────────────────────────
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")

        print(f"\n[Pipeline] 처리 시작: {image_path}")
        print(f"  이미지 크기: {image.shape[1]}x{image.shape[0]}")

        # ── Step 1: YOLO 탐지 ─────────────────────────────────
        print(f"  [1/3] YOLO26 세그멘테이션...")
        detections = self.detector.detect(image)

        # banner 클래스만 필터링 (text 제외)
        banner_detections = [d for d in detections if d['class_id'] == 0]
        print(f"  탐지된 현수막: {len(banner_detections)}개 "
              f"(전체: {len(detections)}개)")

        results = []

        for i, detection in enumerate(banner_detections):
            print(f"\n  [현수막 #{i+1}] 신뢰도: {detection['confidence']:.3f}")

            # ── Step 2: H-matrix 보정 ─────────────────────────
            print(f"    [2/3] H-matrix 원근 보정...")
            corrected, H = self.corrector.correct(
                image,
                detection['polygon']
            )

            if corrected is None:
                print("    [WARN] 원근 보정 실패 (꼭짓점 추출 불가)")
                # 보정 실패시 bounding box crop으로 대체
                x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
                corrected = image[y1:y2, x1:x2]
                H = np.eye(3)  # 단위 행렬 (변환 없음)

            print(f"    보정 이미지 크기: {corrected.shape[1]}x{corrected.shape[0]}")

            # ── Step 3: VLM 분류 ──────────────────────────────
            classification = None
            if self.classifier is not None:
                print(f"    [3/3] VLM 분류...")
                # 첫 탐지 한 번만 디버그 출력(이미지 토큰·픽셀 shape 확인)
                classification = self.classifier.classify(corrected, debug=(i == 0))
                print(f"    결과: {classification['category']} "
                      f"(신뢰도: {classification['confidence']:.2f})")
                if classification.get('extracted_text'):
                    print(f"    문구: {classification['extracted_text'][:80]}")
                print(f"    근거: {classification['reason'][:100]}...")
            else:
                print(f"    [3/3] VLM 미사용 (탐지만 수행)")

            result = {
                'detection':       detection,
                'corrected':       corrected,
                'H_matrix':        H,
                'classification':  classification,
            }
            results.append(result)

            # ── 결과 저장 ─────────────────────────────────────
            if save_dir:
                self._save_result(
                    image, result, i,
                    Path(save_dir),
                    Path(image_path).stem
                )

        # ── 전체 시각화 ───────────────────────────────────────
        if visualize and save_dir:
            self._visualize(image, results, Path(save_dir), Path(image_path).stem)

        return results

    def _save_result(
        self,
        original: np.ndarray,
        result:   dict,
        idx:      int,
        save_dir: Path,
        stem:     str,
    ):
        """
        단일 탐지 결과를 이미지 파일로 저장합니다.

        Args:
            original: 원본 이미지
            result:   파이프라인 결과 딕셔너리
            idx:      현수막 인덱스
            save_dir: 저장 디렉토리
            stem:     파일명 prefix
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        corrected = result['corrected']
        cls = result['classification']

        # 보정된 이미지 저장
        category = cls['category'] if cls else "unknown"
        safe_cat = category.replace(' ', '_')
        out_path = save_dir / f"{stem}_banner{idx:02d}_{safe_cat}.jpg"
        cv2.imwrite(str(out_path), corrected)

    def _visualize(
        self,
        image:    np.ndarray,
        results:  List[dict],
        save_dir: Path,
        stem:     str,
    ):
        """
        탐지 결과를 원본 이미지 위에 시각화합니다.
        발표 자료에 활용할 수 있습니다.

        Args:
            image:    원본 이미지
            results:  파이프라인 결과 목록
            save_dir: 저장 디렉토리
            stem:     파일명 prefix
        """
        vis = image.copy()

        colors = [
            (0, 255, 0),    # 초록 - 정당
            (255, 0, 0),    # 파랑 - 민간
            (0, 0, 255),    # 빨강 - 공공
        ]

        for i, result in enumerate(results):
            det = result['detection']
            cls = result['classification']
            color = colors[i % len(colors)]

            # ── Segmentation mask overlay ─────────────────────
            if 'mask' in det and det['mask'] is not None:
                mask = det['mask']
                # 마스크 크기가 이미지와 다를 수 있으므로 resize
                if mask.shape[:2] != vis.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (vis.shape[1], vis.shape[0])
                    ).astype(bool)
                overlay = vis.copy()
                overlay[mask] = (
                    overlay[mask] * 0.6 +
                    np.array(color) * 0.4
                ).astype(np.uint8)
                vis = overlay

            # ── 꼭짓점 연결선 그리기 ──────────────────────────
            polygon = det['polygon']
            if len(polygon) > 0:
                pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, color, 2)

            # cv2.putText은 Hershey 폰트라 한글 불가 → 영문 라벨로 매핑
            kr_to_en = {
                "정당 현수막": "Party",
                "민간 현수막": "Private",
                "공공 현수막": "Public",
            }
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"#{i+1}"
            if cls:
                en_cat = kr_to_en.get(cls['category'], "Banner")
                label += f" {en_cat} ({cls['confidence']:.2f})"
            else:
                label += f" banner ({det['confidence']:.2f})"

            cv2.rectangle(vis, (x1, y1 - 25), (x1 + len(label) * 9, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        save_path = save_dir / f"{stem}_annotated.jpg"
        cv2.imwrite(str(save_path), vis)
        print(f"  [SAVE] 시각화 저장: {save_path}")


# ============================================================
# CLI 실행
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="현수막 탐지 및 분류 파이프라인"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="처리할 단일 이미지 경로"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="배치 처리할 이미지 디렉토리"
    )
    parser.add_argument(
        "--model",
        default=str(config.YOLO_BEST_MODEL),
        help=f"YOLO 모델 경로"
    )
    parser.add_argument(
        "--output",
        default=str(config.RESULTS_DIR / "pipeline_output"),
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="시각화 결과 저장"
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="VLM 분류 건너뜀 (탐지만 수행)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 신뢰도 임계값"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.batch:
        print("[ERROR] --image 또는 --batch 옵션 중 하나를 지정하세요.")
        print("  예시: python 04_pipeline.py --image sample.jpg --visualize")
        sys.exit(1)

    if args.image and not Path(args.image).exists():
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {args.image}")
        print("  저장소 루트 기준 상대경로 또는 절대경로를 확인하세요.")
        sys.exit(1)

    if args.batch and not Path(args.batch).is_dir():
        print(f"[ERROR] 배치 디렉토리를 찾을 수 없습니다: {args.batch}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"[ERROR] YOLO 가중치가 없습니다: {args.model}")
        print("  아래 중 하나를 수행하세요:")
        print("    1) 학습된 가중치 다운로드:")
        print("       mkdir -p results/models && \\")
        print("       curl -L -o results/models/yolo26_banner_best.pt \\")
        print("         https://github.com/aliceq13/banner-detection-pipeline/releases/download/v1.0/binary2-best.pt")
        print("    2) 직접 학습:  python 02_train_yolo.py")
        sys.exit(1)

    # 파이프라인 초기화
    pipeline = BannerPipeline(
        yolo_model_path=args.model,
        load_vlm=not args.no_vlm,
        yolo_conf=args.conf,
    )

    output_dir = Path(args.output)

    if args.image:
        # 단일 이미지 처리
        results = pipeline.process(
            image_path=args.image,
            visualize=args.visualize,
            save_dir=str(output_dir),
        )

        print("\n" + "=" * 60)
        print("처리 완료")
        print("=" * 60)
        for i, r in enumerate(results):
            cls = r['classification']
            if cls:
                print(f"  현수막 #{i+1}: {cls['category']} "
                      f"(신뢰도: {cls['confidence']:.2f})")
                print(f"    이유: {cls['reason']}")
            else:
                conf = r['detection']['confidence']
                print(f"  현수막 #{i+1}: 탐지됨 (YOLO 신뢰도: {conf:.2f})")

    elif args.batch:
        # 배치 처리
        img_dir = Path(args.batch)
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"[INFO] 배치 처리: {len(images)}장")

        for img_path in images:
            try:
                results = pipeline.process(
                    image_path=str(img_path),
                    visualize=args.visualize,
                    save_dir=str(output_dir),
                )
                print(f"  {img_path.name}: {len(results)}개 현수막 탐지")
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

    print(f"\n결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
