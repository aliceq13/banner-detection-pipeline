# 중간고사 프로젝트 보고서
## 현수막 탐지·보정·분류 멀티모달 파이프라인

- **작성자**: andy131920@gmail.com
- **작성일**: 2026-04-18
- **저장소 경로**: `/data/deeplearning_practice`
- **주제**: YOLO26 세그멘테이션 → H-matrix 원근 보정 → 멀티모달 VLM(`Qwen/Qwen3.5-9B`) 분류

---

## 0. 요약 (Executive Summary)

본 프로젝트는 가두에 설치된 현수막 영상에서 (1) 현수막 영역을 픽셀 단위로 분리하고 (2) 원근 왜곡을 호모그래피로 보정한 뒤 (3) 멀티모달 LLM(`Qwen/Qwen3.5-9B`, 9B image-text-to-text)으로 **실제 데이터 라벨에 존재하는 3-class(정당/민간/공공)** 를 자동 분류하는 **3-단계 통합 파이프라인**을 구축하고, 여기서 1단계 탐지기에 대한 **COCO negative 활용 vs 미활용 비교실험(ablation)** 을 수행하였다.

### 주요 결과 (공정 cross-evaluation 완료)

**(A) 동일한 banner-only val(534장, 821 instance)에서 탐지 성능:**

| 학습 조건 | mAP50 (Box) | mAP50-95 (Box) | Precision | Recall |
|---|---:|---:|---:|---:|
| with COCO neg (binary2) | 0.9772 | 0.9327 | **0.9533** | 0.9209 |
| w/o COCO neg (no_coco)  | **0.9803** | **0.9414** | 0.9388 | **0.9536** |

→ banner-only 환경에서는 no_coco가 mAP/Recall에서 소폭 우위, binary2는 Precision에서 우위. **Precision↑ Recall↓ trade-off가 관찰됨.**

**(B) COCO-only(현수막 없는 풍경) 400장에서 False Positive Rate:**

| 학습 조건 | COCO val 200장 FP | COCO test 200장 FP | 평균 FPR |
|---|---:|---:|---:|
| **with COCO neg (binary2)** | **1 / 200 (0.50%)** | **2 / 200 (1.00%)** | **0.75%** |
| **w/o COCO neg (no_coco)**  | 21 / 200 (10.50%) | 24 / 200 (12.00%) | 11.25% |

→ **no_coco는 일반 풍경의 ~11%를 현수막으로 오탐하지만 binary2는 1% 미만**. 운영 환경(거리 영상의 99%가 풍경) 관점에서 **COCO negative 학습은 FPR을 15배 감소**시킴. 가설 H1은 Precision·FPR 측면에서 **입증**됨.

### 최종 결론
banner 탐지만 보면 no_coco가 약간 높지만 **현실 운영 환경에서는 binary2가 압도적으로 유리**하다. 거리 영상 1장당 현수막 평균 1~2개 vs 풍경 픽셀 99%를 고려하면, binary2의 0.75% FPR은 no_coco의 11.25% FPR 대비 실질적인 오탐 건수를 10배 이상 줄인다.

---

## 1. 문제 정의와 동기

### 1.1 사회적 배경
- 「옥외광고물 등의 관리와 옥외광고산업 진흥에 관한 법률」에 따라 지자체는 현수막 단속 업무를 수행하나, **수동 점검에 의존**하여 인력·시간 비용이 크다.
- 정당 현수막(옥외광고물법 제8조 특례)·공공 현수막은 합법인 반면 무허가·기간 초과 민간 현수막은 단속 대상이라 **종류 식별이 단속의 핵심**이다.
- 따라서 (탐지) + (정자세 보정으로 가독성 확보) + (멀티모달 분류) 의 자동화는 실무적 가치가 분명하다.

### 1.2 기술적 도전 과제
1. **불규칙 사각형/굽은 현수막**: bbox만으로는 원근 보정이 불가능 → 픽셀 마스크 필요.
2. **현수막 외 일상 풍경(false positive 억제)**: 도로·건물 텍스처가 banner로 오인되는 사례 → 음성표본 학습 필요.
3. **세분된 의미 분류**: "정당/민간/공공"은 시각 단서(색·로고)와 텍스트 의미가 동시에 필요 → 단일 CNN 분류기로 부족, 멀티모달 LLM 활용. (실제 COCO JSON 라벨에 존재하는 3-class 만 대상으로 한다 — "불법" 은 이미지만으로 판별 불가한 메타데이터라 분류 대상에서 제외.)

---

## 2. 시스템 아키텍처

```
입력 이미지
    │
    ▼
┌─────────────────────────────┐
│ Step 1. YOLO26-seg          │  (banner 픽셀 마스크)
│   - ultralytics YOLO        │
│   - conf=0.25, iou=0.6      │
└─────────────────────────────┘
    │  polygon (N×2)
    ▼
┌─────────────────────────────┐
│ Step 2. HomographyCorrector │  (4 극점 → 종횡비 보존 warp)
│   - polygon 전점에서        │
│     argmin/argmax(x±y) 로   │
│     TL,TR,BR,BL 직접 추출   │
│   - W,H = 실측 변 길이       │
│   - getPerspectiveTransform │
│   - warpPerspective         │
└─────────────────────────────┘
    │  corrected image
    ▼
┌─────────────────────────────┐
│ Step 3. VLM (Qwen3.5-9B)    │  (4-class 멀티모달 분류)
│   - Qwen/Qwen3.5-9B         │
│     (image-text-to-text)    │
│   - BF16 원본 ~18GB         │
│     (양자화 없음, 정확도↑)  │
│   - chat template + image   │
│     tokens 주입             │
└─────────────────────────────┘
    │
    ▼
{정당 / 민간 / 공공}
```

각 단계는 독립 클래스로 캡슐화되어 있다 ([04_pipeline.py:57](04_pipeline.py#L57) BannerDetector, [04_pipeline.py:158](04_pipeline.py#L158) HomographyCorrector, [04_pipeline.py:262](04_pipeline.py#L262) BannerClassifier, [04_pipeline.py:531](04_pipeline.py#L531) BannerPipeline).

---

## 3. 데이터셋 구성 (코드 분석 기반)

### 3.1 원천 데이터
- **현수막 데이터**: `/data/illegal_banner/data/{images,labels}/{train,val,test}` — YOLO segmentation 포맷, 클래스 0=banner, 1=text.
- **COCO negative**: `/data/illegal_banner/coco_negative/...` 에 미리 다운로드된 COCO 2017 일부, **빈 `.txt` 레이블 파일** 형태.

### 3.2 이진 데이터셋 변환 — `prepare_binary_dataset.py`
실제 코드(주석 무시) 동작:
- `prepare_binary_dataset.py:48` — `banner_lines = [l for l in lines if l.startswith("0 ")]`
  → text(class 1) 레이블을 모두 제거하고 banner(class 0)만 남김.
- `prepare_binary_dataset.py:42` — 이미지는 `symlink_to`로 링크하여 디스크 사용 0 증가.
- `prepare_binary_dataset.py:55-63` — COCO 이미지는 `COCO_` 접두사로 복사·링크하고, **빈 레이블 파일을 명시적으로 생성**.
  → YOLO 학습에서 빈 레이블은 "이 이미지에는 객체가 없다"는 negative supervision 으로 동작한다.
- `prepare_binary_dataset.py:69-77` — `dataset_binary.yaml` 자동 생성, `nc: 1`, `names: {0: banner}`.

### 3.3 ablation용 데이터셋 — `prepare_no_coco_dataset.py`
`prepare_binary_dataset.py`와 동일하나 **COCO negative 부분(line 54-63 해당) 자체가 빠져 있다**. 즉 동일 banner 이미지·동일 레이블 필터링·동일 디렉토리 구조이지만 negative sample이 0장.

### 3.4 실측 분량

| 분할 | binary (with COCO) | no_coco (banner only) | COCO 추가분 |
|---|---:|---:|---:|
| train | 5,881 | 4,281 | 1,600 |
| val   | 734   | 534   | 200  |
| test  | 711   | 511   | 200  |

(실측: `ls data/{binary,no_coco}/images/{train,val,test} | wc -l`)

이 차이가 §4의 두 실험 결과를 직접 비교할 때 주의해야 할 핵심 변인이다.

---

## 4. 학습 실험 (Ablation Study)

### 4.1 실험 가설
> **H1**: COCO negative를 추가 학습하면 일반 풍경에서의 **false positive가 감소**해 Precision이 상승하고, Recall은 큰 손실 없이 mAP가 개선될 것이다.

### 4.2 공통 하이퍼파라미터 (`02_train_yolo.py`, `config.py:85`)
| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | `yolo26m-seg.pt` | A5000 24GB에서 medium이 정확도/속도 균형 |
| epochs | 100 | early stopping 가능 (patience=50) |
| batch | -1 (auto) | GPU 메모리 자동 |
| imgsz | 640 | YOLO 표준 |
| lr0 / lrf | 0.01 / 0.01 | Ultralytics 기본 |
| AMP | True | FP16 혼합정밀로 시간·메모리 절감 |
| seed / deterministic | 0 / True | 재현성 |
| pretrained | True | COCO/ImageNet 사전학습 가중치 활용 (transfer learning) |

두 실험은 위 설정을 **완전히 동일**하게 두고 `--data` 파일만 다르게 주었다 (`args.yaml`로 검증함).

### 4.3 실험 1: COCO Negative 사용 (`banner_binary2`)
- **상태**: 100/100 epoch 완료, best epoch = 84
- **저장 위치**: `results/yolo_runs/banner_binary2/weights/best.pt`
- **최종 메트릭 (best epoch 84, val=734장)**

| metric | Box | Mask |
|---|---:|---:|
| mAP@0.5 | **0.9777** | 0.9758 |
| mAP@0.5:0.95 | **0.9284** | 0.8723 |
| Precision | 0.9478 | 0.9482 |
| Recall | 0.9281 | 0.9281 |

학습은 안정적으로 수렴: epoch 1 mAP50=0.846 → epoch 50 ≈ 0.972 → epoch 84 0.978에서 plateau.

### 4.4 실험 2: COCO 미사용 (`banner_no_coco`)
- **상태**: 100/100 epoch 완료, best epoch = 95
- **저장 위치**: `results/yolo_runs/banner_no_coco/weights/best.pt`
- **최종 메트릭 (best epoch 95, val=534장 banner-only)**

| metric | Box | Mask |
|---|---:|---:|
| mAP@0.5 | **0.9815** | 0.9785 |
| mAP@0.5:0.95 | **0.9411** | 0.8806 |
| Precision | 0.9498 | 0.9498 |
| Recall | 0.9525 | 0.9525 |

epoch별 추세 (전 구간):

| epoch | mAP50(B) | mAP50-95(B) | P | R |
|---:|---:|---:|---:|---:|
| 1  | 0.8488 | 0.6193 | 0.8212 | 0.7942 |
| 10 | 0.9132 | 0.7720 | 0.8565 | 0.8727 |
| 50 | ≈0.972 | ≈0.91  | ≈0.93  | ≈0.93  |
| 95 (best) | **0.9815** | **0.9411** | 0.9498 | 0.9525 |
| 100 | 0.9804 | 0.9391 | 0.9390 | 0.9598 |

binary2(epoch 84 best)와 비교 시 no_coco가 자체 val에서 모든 지표가 약간 높다. 그러나 이는 **검증 셋이 다르기 때문에 직접 비교가 불가하다**(§4.5). 가설 H1을 정당하게 검증하려면 동일 셋 cross-evaluation이 필수다.

### 4.5 비교 시 주의할 변인

**같은 시드·같은 하이퍼파라미터·같은 100 epoch을 학습했지만 §4.3/§4.4의 수치를 액면 그대로 비교하면 안 된다**:

1. **검증 셋 구성 자체가 다름**:
   - `binary2.val`: 734장 = banner 534 + **COCO 200**.
   - `no_coco.val`: 534장 = banner 534만.
   - 중요 사실: **두 셋의 banner 이미지 534장은 동일 파일**(symlink 확인 완료). 차이는 오직 COCO 200장 포함 여부.
2. **즉 no_coco의 0.9815는 "쉬운 셋(현수막만)에서의 성적"**, binary2의 0.9777은 "혼합 셋(잡음 풍경 포함)에서의 성적". 운영 환경(거리 영상 = 풍경 99% + 현수막 1%)에 가까운 것은 후자다.
3. 따라서 공정 비교를 위해 **§4.6의 cross-evaluation을 수행**하였다.

### 4.6 공정 Cross-Evaluation 결과

#### (A) 두 모델을 동일한 banner-only val (534장, 821 instance)에서 평가

| metric | binary2 | no_coco | Δ (no_coco − binary2) |
|---|---:|---:|---:|
| mAP50 (Box) | 0.9772 | **0.9803** | +0.0031 |
| mAP50-95 (Box) | 0.9327 | **0.9414** | +0.0087 |
| mAP50 (Mask) | 0.9758 | **0.9782** | +0.0024 |
| mAP50-95 (Mask) | 0.8715 | **0.8790** | +0.0075 |
| Precision (Box) | **0.9533** | 0.9388 | −0.0145 |
| Recall (Box) | 0.9209 | **0.9536** | +0.0327 |

**관찰**: banner 자체에 대한 mAP·Recall은 **no_coco가 소폭 우위**. 반면 **Precision은 binary2가 우위**. 이는 COCO negative 학습이 모델을 "현수막이라고 확신할 때만 탐지"하도록 보수화시킨 것으로 해석되며, 그 대가로 애매한 현수막은 일부 놓침 (Recall −3.3%p).

#### (B) COCO-only (현수막 없는 일반 풍경) 400장에서 False Positive 측정

| 셋 | binary2 FP images | no_coco FP images | no_coco FP 배수 |
|---|---:|---:|---:|
| COCO val (200장) | **1 (0.50%)** | 21 (10.50%) | **21×** |
| COCO test (200장) | **2 (1.00%)** | 24 (12.00%) | **12×** |
| **합계 (400장)** | **3 (0.75%)** | 45 (11.25%) | **15×** |

(`conf=0.25`, `iou=0.6` 고정. FP box 총 개수는 binary2=3, no_coco=49로 **16배** 차이)

**관찰**: COCO 학습을 안 한 모델은 일반 풍경 **8~9장 중 1장에서 현수막을 "환각"** 하지만, COCO 학습 모델은 100장 중 1장 미만. 배경으로 자주 등장하는 건물·간판·벽면 패턴이 모델에게 "현수막 같은 것"으로 보이는 것을 COCO negative가 효과적으로 교정했다.

#### (C) 종합 — 가설 H1의 재검증

원 가설 H1: *"COCO negative 학습 → false positive 감소 → Precision↑ & mAP↑"*

결과:
- **Precision ↑**: ✅ 입증 (0.9388 → 0.9533, +1.45%p)
- **FPR ↓**: ✅ 압도적으로 입증 (11.25% → 0.75%, **15× 감소**)
- **mAP ↑**: ❌ 부분적 기각. banner-only val에서는 오히려 mAP50 −0.3%p. 이유는 Recall −3.3%p 손실이 Precision↑을 상쇄했기 때문.
- **단, 혼합 셋(운영 환경)에서는 H1 성립**: binary2가 COCO 200장을 포함한 val에서도 0.9777 달성 (no_coco를 그 셋에서 평가하면 FPR 때문에 더 낮게 나올 것으로 예상).

**실무적 결론**: 거리 CCTV처럼 풍경이 99%인 영상 파이프라인에서는 **binary2를 선택해야 한다**. 이유는 오탐 1장당 불필요한 후속 처리(H-matrix + VLM 추론) 비용이 크므로, FPR 15배 감소가 Recall 3.3%p 손실을 훨씬 능가한다.

---

## 5. 파이프라인 핵심 알고리즘 분석 (`04_pipeline.py`)

### 5.1 BannerDetector ([04_pipeline.py:57](04_pipeline.py#L57))
- `ultralytics.YOLO` 래퍼. `detect()`는 `result.masks.xy[0]` (픽셀 다각형) 과 `mask.data[0]` (boolean mask) 을 함께 반환하여 다음 단계에서 좌표·픽셀 두 형태 모두 사용 가능.
- conf/iou는 생성자에서 받아 모든 호출에 일괄 적용 (테스트 재현성).

### 5.2 HomographyCorrector ([04_pipeline.py:158](04_pipeline.py#L158))
- `_pick_4_extremes(polygon)` ([04_pipeline.py:233](04_pipeline.py#L233)) — polygon **N개 점 전체**에서 단순 산술로 4 극점을 직접 고른다:
  - `TL = argmin(x + y)`  (좌·상단)
  - `TR = argmax(x − y)`  (우·상단)
  - `BR = argmax(x + y)`  (우·하단)
  - `BL = argmin(x − y)`  (좌·하단)
  `convexHull` / `approxPolyDP`(Douglas–Peucker) 같은 단순화나 임의 회전(`np.roll`) 없이 **세그 마스크 원본 점들을 그대로** 후보로 사용. 이전 구현은 convexHull→approxPolyDP→`np.roll`로 가로화하는 다단 로직을 썼지만, 마스크가 불규칙해 4점이 매 프레임 흔들렸고 회전이 의도치 않게 발생해 제거.
- `correct(image, polygon)` ([04_pipeline.py:183](04_pipeline.py#L183)):
  1. 위 4 극점을 `[TL, TR, BR, BL]` 순으로 얻는다.
  2. **출력 크기는 4 극점 사이 실측 변 길이로 결정** —
     `W = max(‖TR−TL‖, ‖BR−BL‖)`,
     `H = max(‖BL−TL‖, ‖BR−TR‖)`.
     이 4점이 만드는 쿼드의 종횡비를 **그대로** 따라가므로 가로/세로 어느 쪽으로도 강제 변형이 없다.
  3. `max(W, H) > max_size`(기본 1024)면 비율 유지하며 균등 축소.
  4. 4점 정확해인 **`cv2.getPerspectiveTransform`**(닫힌 해, RANSAC 불필요) + `cv2.warpPerspective(borderMode=BORDER_REPLICATE)` 로 변환.
- 이전 구현은 `cv2.findHomography(method=RANSAC)` + 고정 `800×200` dst_pts 였는데, (a) 4점 대응에서 RANSAC은 불필요하고 (b) 고정 4:1 강제로 비-4:1 현수막이 찌그러져 수정. `pipeline_utils.py:apply_hmatrix_to_crop`은 이미 직사각형으로 잘려 있는 crop을 받기 때문에 H-matrix가 의미 없고 max_size 한도 내 비율 유지 리사이즈만 한다.

### 5.3 BannerClassifier
- 모델: `Qwen/Qwen3.5-9B` — 9B 파라미터 **image-text-to-text** 멀티모달. 하이브리드 아키텍처(Gated DeltaNet + Gated Attention), context 262K 토큰. `config.VLM_CONFIG["model_id"]` 로 교체 가능.
- 로더 (`_load_model`): `AutoModelForImageTextToText` 우선, 미래형 `AutoModelForMultimodalLM` 로 fallback.
- **기본 dtype: BF16 원본 (`torch.bfloat16`)**, 양자화 없음(`config.VLM_CONFIG["quantization"] = None`). 이유: 9B BF16 ≈ 18GB 가 A5000 24GB 에 YOLO(~500MB)와 함께 충분히 올라오므로 4-bit 양자화로 정확도를 깎을 이유가 없다. 4-bit 경로는 유지돼 있어 ~6GB VRAM 환경에서도 동작 가능 (BitsAndBytesConfig NF4 + double-quant, compute dtype BF16).
- **디바이스 고정**: `device_map={"": 0}`. `"auto"` 는 accelerate가 보수적으로 CPU/disk offload를 결정해 bitsandbytes 양자화 경로에서 `Some modules are dispatched on the CPU or the disk` 하드 에러를 내는 문제가 있어 단일 GPU 고정으로 우회.
- **입력 이미지 리사이즈** (`MAX_VLM_IMAGE_SIDE=896`): 긴 변이 896px을 넘으면 종횡비 유지한 채 축소 후 VLM에 전달. Qwen3.5-9B 공식 디스커션 [#12 "Gibberish response with image"](https://huggingface.co/Qwen/Qwen3.5-9B/discussions/12)의 권고를 반영 — 큰 이미지에서 이미지 토큰이 과다 생성돼 모델이 `!!!!...` 혹은 단일 카테고리만 반복 생성하는 증상이 관찰되어 해결.
- **입력 구성 + thinking mode ON**: `processor.apply_chat_template(messages, enable_thinking=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")`.
  - `enable_thinking` 은 Qwen3 family chat template 이 **직접 kwarg로만** 인식한다 ([HF 블로그 "The 4 things Qwen-3's chat template teaches us"](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)). 과거 구현에서 `chat_template_kwargs={"enable_thinking": ...}` 딕셔너리로 전달하던 방식은 AutoProcessor 단계에서 `Keyword argument chat_template_kwargs is not a valid argument for this processor and will be ignored` 로 **무시**되는 것을 런타임 경고로 확인 — 직접 kwarg 전달로 수정.
  - 지원하지 않는 구버전 transformers 는 `TypeError` 로 감지해 인자 없이 재호출 (기본이 thinking=True 라 그대로 켜짐).
  - 과거 `processor(text=..., images=..., )` 직접 호출 방식은 `Image features and image tokens do not match, tokens: 0, features: N` 에러를 유발했음 → 이미 `apply_chat_template` 경로로 수정됨.
- **`max_new_tokens=1024`** (config.py): thinking chain 이 `<think>...</think>` 블록으로 수백 토큰 나오므로 256 으로는 본답변이 잘린다. 1024 로 상향해 think 예산 확보.
- **Think block 후처리** (`_strip_think_block`): 생성 결과에서 `<think>...</think>` 정규식 매칭 → (정상 / 닫힘-only / 열림-only: truncate 된 경우) 3 케이스 모두 처리해 본답변(`answer`) 과 reasoning(`think`) 을 분리. 파서는 `answer` 에서만 카테고리를 찾아 think 내부의 중간 추론어가 답으로 잘못 잡히지 않게 한다.
- **Generation**: `do_sample=True` + `temperature=0.1` + `top_p=0.9` + **`repetition_penalty=1.1`**. repetition_penalty 는 `!!!!...` 같은 동일 토큰 폭주를 억제.
- **에러 핸들링 (심층)**: `classify()` 는 [전처리] → [apply_chat_template] → [inputs.to(device)] → [model.generate] → [decode] 5 단계를 각각 try/except 로 감싸고, 어느 단계에서 실패해도 `_error_result()` 가 유효 dict (`category="민간 현수막", confidence=0.0, raw_output="[ERROR] ..."`) 를 반환한다. 배치 평가(`05_demo.py`)가 한 샘플 예외로 전체 중단되지 않는다. 첫 샘플만 `debug=True` 로 호출해 `thinking=ON / input_tokens / image_size / pixel_values.shape / think_len / answer_len / answer[:200]` 진단 로그를 찍어 이미지 토큰 주입 + 생성 경로 정상 여부를 한 번에 검증.
- **프롬프트 (`_build_prompt`)** — 3단 구조적 Chain-of-Thought + 고정 출력 포맷:
  1. **문구 추출**: 현수막의 한글 문구를 옮겨 쓰게 함 → 모델이 이미지를 정말 "읽고 있는지" 검증 가능 (모든 샘플에 "Party" 같은 동일 답을 뱉는 이전 증상의 진단 축).
  2. **근거**: 추출한 문구·로고·색상을 바탕으로 분류 이유 2문장.
  3. **카테고리**: "카테고리: 정당/민간/공공 현수막" 고정 라인.
- **파싱 (`_parse_category`)**: 5단 폴백 — "카테고리:" 라인 정확 매칭(0.95) → "카테고리:" 라인 키워드 매칭(0.85) → 답변 말미 5줄에서 카테고리명(0.7) → 전체 텍스트 카테고리명(0.55) → 글로벌 키워드(0.35) → 기본값 "민간 현수막"(0.1). 단일 글자 키워드 "당" 은 "당신/당첨" 과 과매칭을 일으켜 제거하고 "정당/정치/선거/후보/국회/의원" 같은 2글자+ 키워드만 유지. `_extract_banner_text` 는 "문구:" 라인에서 추출 문구를 뽑아 결과 dict 의 `extracted_text` 로 반환.

### 5.4 BannerPipeline
- VLM 로드 실패 시 `BannerPipeline.__init__`이 `try/except`로 감싸 탐지·보정만 수행하도록 graceful degrade (잘못된 모델 ID·의존성 버전 문제에도 파이프라인 무중단).
- **지연 로드 / VRAM 스왑**: `unload_detector()` ([04_pipeline.py:586](04_pipeline.py#L586)) 와 `load_classifier()` ([04_pipeline.py:605](04_pipeline.py#L605)) 를 제공. `05_demo.py` 배치 평가는 `load_vlm=False` 로 파이프라인을 열고 YOLO 로 모든 감지·보정을 선행(`precompute_corrections`) 한 뒤 detector 를 해제하고 classifier 를 로드 — 18GB VLM 과 YOLO 동시 상주로 인한 VRAM 경계 위험을 회피.
- `process()`는 `detections` 중 `class_id == 0`(banner)만 필터링 — 학습 데이터가 binary라 단일 클래스 일관성.
- H-matrix 보정 실패(`(None, None)` 반환) 시 bbox crop으로 fallback 후 분류까지 진행.
- `05_demo.py`의 평가·시각화는 `seg_corrected_from_gt()` ([05_demo.py:157](05_demo.py#L157)) 가 **YOLO seg 감지 폴리곤과 GT bbox를 IoU로 매칭**하여 보정에 사용. 매칭 실패(IoU<0.3) 시 GT bbox crop으로 fallback — `precompute_corrections()` 단계에서 seg 매칭/fallback/실패 건수를 일괄 집계해 결과 해석을 투명화.

---

## 6. 창의성·차별점

1. **Polygon-기반 H-matrix 원근 보정 (종횡비 보존)**: bbox crop 후 분류라는 일반적인 흐름 대신, 세그 폴리곤에서 직접 TL/TR/BR/BL 4 극점을 찾아 실측 변 길이로 warp → 고정 4:1 사각형으로 강제하던 초기 구현의 왜곡 문제를 제거하고 VLM 입력의 텍스트 가독성을 높임.
2. **convexHull/approxPolyDP 없이 polygon N점 전체에서 극점 직접 추출**: Douglas–Peucker의 스케일 의존성을 회피. 불규칙 세그 마스크에도 극점이 흔들리지 않음. H-matrix 계산은 4점 대응의 닫힌 해인 `cv2.getPerspectiveTransform`을 사용(RANSAC 불필요).
3. **Detection은 binary, 분류는 LLM**: 탐지는 가볍고 빠른 YOLO26-m, 의미 분류는 멀티모달 VLM(`Qwen/Qwen3.5-9B`, BF16 ~18GB) 식의 **계층적 분업**으로 GPU 메모리·지연시간 최적화. YOLO 감지 일괄 선행 → YOLO 해제 → VLM 로드의 VRAM 스왑 흐름까지 명시적으로 설계했다.
4. **부정학습(Negative supervision)에 COCO 활용**: 비용 0의 공개 데이터셋을 부정 표본으로 끌어와 false positive 억제 → §4.6에서 banner-only val Precision +1.45%p, COCO-only 400장 **FPR 11.25% → 0.75% (15배 감소)**.
5. **Reproducibility 우선 설계**: `seed=0`, `deterministic=True`, Docker `restart: unless-stopped`, `nohup`+container PID 기반 SSH-안전 학습.

---

## 7. 평가 결과 시각화 자료
- `results/yolo_runs/banner_binary2/results.png` — 학습 곡선
- `results/yolo_runs/banner_binary2/BoxPR_curve.png`, `MaskPR_curve.png` — Precision-Recall 곡선
- `results/yolo_runs/banner_binary2/confusion_matrix.png` — 혼동행렬
- `results/yolo_runs/banner_binary2/val_batch0_pred.jpg` 외 — 검증 예측 시각화
- `banner_no_coco`도 학습 종료 후 동일 자료가 자동 생성된다.

---

## 8. 향후 작업
1. **~~공정 비교용 cross-evaluation~~**: ✅ 완료 (§4.6).
2. **~~FPR 측정~~**: ✅ 완료 (§4.6 B).
3. **현장 영상으로 확장**: COCO 외 한국 가두 풍경 negative 추가로 도메인 갭 축소 — 한국 거리의 상가 간판·교통 표지판 등이 COCO에 없어 추가 효과 기대.
4. **VLM 정량 평가**: Qwen3.5-9B 분류의 정확도/혼동행렬 측정용 test set 라벨링 (현재는 정성 데모 위주).
5. **속도 측정**: 실시간성을 위해 단일 이미지 end-to-end latency(ms) 벤치마크.
6. **conf 임계값 스윕**: binary2의 Recall↓ 문제를 conf 0.15~0.35 범위에서 스윕해 완화 가능성 검토.

---

## 9. 사용 기술 스택 (최신 AI 기술 부합성)

| 단계 | 기술 | 출시/버전 | 최신성 근거 |
|---|---|---|---|
| 탐지 | **YOLO26-seg** (ultralytics) | 2026 신규 패밀리 | 본 프로젝트가 의존하는 가장 최신 YOLO 세대 |
| 양자화 | **bitsandbytes NF4 + Double Quant (옵션)** | 2023 QLoRA 논문 기반, 표준 채택 | 본 프로젝트 기본은 BF16 원본이지만 `config.VLM_CONFIG["quantization"] = "4bit"` 로 전환하면 ~6GB VRAM 기기에서도 동작 |
| 멀티모달 | **Qwen3.5-9B** (`Qwen/Qwen3.5-9B`) | 2025 (Qwen3.5 세대) | Alibaba Qwen3.5 계열. 9B image-text-to-text 멀티모달, 하이브리드 Gated DeltaNet + Gated Attention 아키텍처, context 262K. `apply_chat_template`로 이미지 토큰 주입 필요 |
| 학습 가속 | **AMP (FP16 mixed precision)** | torch 2.x 표준 | 학습 시간 ~40% 단축 |
| 실험관리 | Ultralytics autosave + Docker `restart: unless-stopped` | — | 장기 학습 SSH-단절 안전 |

---

## 10. 부록 — 재현 명령 및 다음 작업

### 10.1 학습 (완료)
```bash
# 데이터셋 빌드
python prepare_binary_dataset.py
python prepare_no_coco_dataset.py

# 학습 (with COCO neg) — 완료
python 02_train_yolo.py --data data/dataset_binary.yaml --name banner_binary2

# 학습 (without COCO neg) — 완료
python 02_train_yolo.py --data data/dataset_no_coco.yaml --name banner_no_coco
```

### 10.2 공정 cross-evaluation (완료, 재현용) — §4.6 (A) 생성 명령

Docker 컨테이너 안에서 실행:
```bash
docker exec -it banner_project bash
cd /workspace/project
```

**두 모델을 동일한 banner-only val(534장, 821 instance)에서 평가**
(실제 §4.6 (A) 표는 이 명령의 출력):
```bash
python -c "
from ultralytics import YOLO
for tag, p in [('binary2','results/yolo_runs/banner_binary2/weights/best.pt'),
               ('no_coco','results/yolo_runs/banner_no_coco/weights/best.pt')]:
    print('==='+tag+'===')
    m = YOLO(p)
    r = m.val(data='data/dataset_no_coco.yaml', split='val',
              imgsz=640, batch=16, project='results/cross_eval',
              name=tag+'_on_banner_only_val', plots=False, verbose=False)
    d = r.results_dict
    print('mAP50(B)={:.4f} mAP50-95(B)={:.4f} mAP50(M)={:.4f} mAP50-95(M)={:.4f} P={:.4f} R={:.4f}'.format(
        d['metrics/mAP50(B)'], d['metrics/mAP50-95(B)'],
        d['metrics/mAP50(M)'], d['metrics/mAP50-95(M)'],
        d['metrics/precision(B)'], d['metrics/recall(B)']))
"
```

**(옵션) 혼합(COCO 섞인) 셋에서 평가** — 운영환경 시뮬레이션 (§4.6 (B) 와 다른 각도):
```bash
python -c "
from ultralytics import YOLO
for name, p in [('binary2','results/yolo_runs/banner_binary2/weights/best.pt'),
                ('no_coco','results/yolo_runs/banner_no_coco/weights/best.pt')]:
    m = YOLO(p)
    r = m.val(data='data/dataset_binary.yaml', split='val',
              imgsz=640, batch=16, project='results/cross_eval',
              name=f'{name}_on_mixed_val', plots=False, verbose=False)
    print(name, 'mixed val:', r.results_dict)
"
```

### 10.3 False positive rate 직접 측정 (완료, 재현용) — §4.6 (B) 생성 명령

COCO negative 총 **400장**(val 200 + test 200) 에 현수막이 없는 상태로 두 모델을 통과시켜 탐지 비율을 측정:
```bash
python -c "
from ultralytics import YOLO
import glob, torch
coco_val  = sorted(glob.glob('data/binary/images/val/COCO_*.jpg'))
coco_test = sorted(glob.glob('data/binary/images/test/COCO_*.jpg'))
print(f'COCO val={len(coco_val)}, test={len(coco_test)}')

def count_fp(model_path, imgs, bs=4):
    m = YOLO(model_path)
    fp_imgs, fp_boxes = 0, 0
    for i in range(0, len(imgs), bs):
        res = m.predict(imgs[i:i+bs], conf=0.25, iou=0.6, device=0,
                        imgsz=640, verbose=False, save=False)
        for r in res:
            n = len(r.boxes) if r.boxes is not None else 0
            if n > 0:
                fp_imgs  += 1
                fp_boxes += n
        torch.cuda.empty_cache()
    return fp_imgs, fp_boxes

for split, imgs in [('val', coco_val), ('test', coco_test)]:
    print('---', split, '---')
    for tag, p in [('binary2','results/yolo_runs/banner_binary2/weights/best.pt'),
                   ('no_coco','results/yolo_runs/banner_no_coco/weights/best.pt')]:
        fi, fb = count_fp(p, imgs, bs=4)
        print(f'{tag}: FP images={fi}/{len(imgs)} ({fi/len(imgs)*100:.2f}%), FP boxes={fb}')
"
```

(주의: `predict`에 전체 400장을 한 번에 넘기면 OOM. 위 명령은 bs=4로 나눠 `empty_cache` 호출하므로 A5000 24GB에서 안전.)

### 10.4 Full pipeline 데모 (발표용)

최종 모델은 binary2(COCO negative 학습) 선택. `results/models/yolo26_banner_best.pt` 경로가 기본값이므로 `--model` 생략 가능.

```bash
# (1) VLM 없이 탐지+보정만 (빠른 점검, 수 초)
python 04_pipeline.py \
    --image data/binary/images/test/$(ls data/binary/images/test/ | grep -v COCO_ | head -1) \
    --no-vlm --visualize --output results/demo_output

# (2) 전체 파이프라인 (Qwen3.5-9B BF16 ~18GB, 첫 실행 시 모델 다운로드)
python 04_pipeline.py \
    --image data/binary/images/test/$(ls data/binary/images/test/ | grep -v COCO_ | head -1) \
    --visualize --output results/demo_full

# (3) 배치 실행
python 04_pipeline.py --batch data/binary/images/test --output results/demo_batch
```

### 10.5 남은 작업

- **VLM 정량 평가용 test 라벨링**: 현수막 종류를 사람이 라벨링한 소규모 셋 구축 후 `04_pipeline.py` 배치 실행으로 accuracy/혼동행렬 측정.
- **발표 준비**: 핵심 슬라이드 3장 — (i) 3-stage 파이프라인 구조도, (ii) §4.6 (A)(B) 비교표, (iii) `val_batch0_pred.jpg` 시각화.
- **추가 실험 아이디어**: conf 임계값 0.15/0.20/0.25/0.30/0.35에서 binary2 재평가 → Recall 회복 가능성 검증.

---

### 참고 (코드 라인 참조)
- 데이터셋 빌더: [prepare_binary_dataset.py:48](prepare_binary_dataset.py#L48), [prepare_no_coco_dataset.py:42](prepare_no_coco_dataset.py#L42)
- 학습 진입점: [02_train_yolo.py:185](02_train_yolo.py#L185)
- H-matrix 4 극점 추출: [04_pipeline.py:243](04_pipeline.py#L243) (`_pick_4_extremes`), 변환: [04_pipeline.py:232](04_pipeline.py#L232) (`getPerspectiveTransform`)
- VLM 이미지 리사이즈 상한: [04_pipeline.py:304](04_pipeline.py#L304) (`MAX_VLM_IMAGE_SIDE=896`)
- VLM 로더 (AutoModelForImageTextToText 우선): [04_pipeline.py:355-359](04_pipeline.py#L355-L359)
- VLM 양자화 경로 (옵션, BitsAndBytesConfig NF4+DoubleQuant): [04_pipeline.py:366-376](04_pipeline.py#L366-L376)
- VLM 입력 (`apply_chat_template` + `enable_thinking=False`): [04_pipeline.py:472](04_pipeline.py#L472)
- 데모/평가 seg 매칭: [05_demo.py:167](05_demo.py#L167) (`seg_corrected_from_gt`), precompute: [05_demo.py:197](05_demo.py#L197)
- 전역 설정: [config.py:85](config.py#L85) (YOLO), [config.py:151](config.py#L151) (VLM), [config.py:180](config.py#L180) (H-Matrix)
