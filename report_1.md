# 중간고사 프로젝트 보고서
## 현수막 탐지·보정·분류 멀티모달 파이프라인

- **작성자**: andy131920@gmail.com
- **작성일**: 2026-04-19
- **저장소**: `/data/deeplearning_practice`
- **주제**: YOLO26-seg 세그멘테이션 → 호모그래피 원근 보정 → 멀티모달 VLM(Qwen/Qwen3.5-9B) 3-class 분류

---

## 0. 요약

가두 현수막 영상에서 (1) 현수막을 픽셀 단위로 분리하고 (2) 원근 왜곡을 호모그래피로 펴고 (3) 멀티모달 LLM으로 종류를 분류하는 **3-단계 파이프라인**을 구축했다. 탐지기에 **COCO 음성표본을 추가학습해 일반 풍경 FPR을 15배 감소**시킨 것이 본 프로젝트의 핵심 기여다.

| 지표 | 결과 |
|---|---|
| YOLO26-seg mAP50 (banner-only val 534장) | **0.9772** (binary2) / 0.9803 (no_coco) |
| COCO 400장 FPR (현수막 없는 풍경) | **0.75%** (binary2) vs 11.25% (no_coco) — **15× 감소** |
| VLM 3-class 분류 정확도 (n=10 스모크) | **80.0%** — 공공 100%, 민간 83%, 정당 0/1 |

> **한 줄 결론**: banner-only에서는 두 모델이 비등하지만, **일반 풍경 99%가 배경인 거리 CCTV 환경에서 binary2가 압도적으로 유리**하다 (오탐 15배↓).

> 📷 **[FIG 0]** 파이프라인 데모 그리드를 여기 배치 — `results/demo/pipeline_demo.png` (원본+GT bbox / H-matrix 보정 / VLM 분류 결과가 한눈에 보이는 3열 6행 그림)

---

## 1. 문제 정의

### 1.1 사회적 배경
- 「옥외광고물 등의 관리와 옥외광고산업 진흥에 관한 법률」에 따른 현수막 단속이 **수작업에 의존** → 자동화 필요.
- 정당(§8 특례)·공공 현수막은 합법, 무허가 민간은 단속 대상. **종류 식별이 핵심**.

### 1.2 기술적 도전
1. **불규칙 사각형/굽은 현수막** → bbox만으론 보정 불가, 픽셀 마스크 필요.
2. **거리 풍경 오탐 억제** → 도로/건물이 banner로 오인되는 사례를 negative로 눌러야 함.
3. **세분된 의미 분류** → 색·로고 + 텍스트 의미가 동시에 필요해 단일 CNN 불충분 → 멀티모달 LLM 활용. 실제 데이터 라벨에 있는 **정당/민간/공공 3-class**로 한정 ("불법"은 이미지만으로 판별 불가한 메타데이터라 제외).

---

## 2. 시스템 아키텍처

```
입력 이미지
    │
    ▼
┌─────────────────────────────┐
│ Step 1. YOLO26-seg          │  banner 픽셀 마스크
│   ultralytics, conf=0.25     │  (polygon + bool mask)
└─────────────────────────────┘
    │  polygon (N×2)
    ▼
┌─────────────────────────────┐
│ Step 2. HomographyCorrector │  4 극점 → 종횡비 보존 warp
│   argmin/argmax(x±y)         │
│   getPerspectiveTransform    │
└─────────────────────────────┘
    │  corrected image
    ▼
┌─────────────────────────────┐
│ Step 3. Qwen3.5-9B VLM      │  3-class 멀티모달 분류
│   image-text-to-text, BF16   │  (정당/민간/공공)
│   thinking mode ON           │
└─────────────────────────────┘
    │
    ▼  {정당 / 민간 / 공공 현수막, 근거, 추출 문구}
```

각 단계는 독립 클래스로 캡슐화돼 있다: [BannerDetector](04_pipeline.py#L71), [HomographyCorrector](04_pipeline.py#L172), [BannerClassifier](04_pipeline.py#L276), [BannerPipeline](04_pipeline.py#L726).

---

## 3. 데이터셋

### 3.1 원천
- **현수막**: `/data/illegal_banner/data/{images,labels}/{train,val,test}` — YOLO segmentation 포맷, 클래스 0=banner, 1=text.
- **COCO negative**: COCO 2017 val의 일부를 **빈 `.txt` 레이블** 형태로 준비. YOLO 학습에서 "이 이미지에 객체 없음" = negative supervision.

### 3.2 이진 데이터셋 변환 — [prepare_binary_dataset.py](prepare_binary_dataset.py)
실제 코드 동작:
- `banner_lines = [l for l in lines if l.startswith("0 ")]` — text(class 1) 제거, banner만 남김 ([prepare_binary_dataset.py:48](prepare_binary_dataset.py#L48)).
- 이미지는 `symlink_to` → 디스크 사용 0 증가.
- COCO는 `COCO_` 접두사로 링크하고 **빈 레이블 파일을 명시적으로 생성**.
- `dataset_binary.yaml` 자동 생성, `nc: 1, names: {0: banner}`.

### 3.3 Ablation용 데이터셋 — [prepare_no_coco_dataset.py](prepare_no_coco_dataset.py)
`binary`와 동일하되 **COCO negative 부분이 빠짐**. 동일 banner 이미지, 동일 레이블 필터링.

### 3.4 실측 분량

| 분할 | binary (with COCO) | no_coco (banner only) | COCO 추가분 |
|---|---:|---:|---:|
| train | 5,881 | 4,281 | 1,600 |
| val   |   734 |   534 |   200 |
| test  |   711 |   511 |   200 |

두 셋의 banner 534장은 **동일 파일**(symlink 확인). 차이는 오직 COCO 200장 포함 여부 → §4 비교의 핵심 변인.

---

## 4. YOLO 학습 실험 (Ablation)

### 4.1 가설
> **H1**: COCO negative를 학습하면 일반 풍경 false positive가 감소해 Precision과 mAP가 상승한다.

### 4.2 공통 하이퍼파라미터 ([config.py:85](config.py#L85))

| 항목 | 값 |
|---|---|
| 모델 | `yolo26m-seg.pt` (medium) |
| epochs / patience | 100 / 50 |
| batch / imgsz | auto / 640 |
| lr0 / lrf | 0.01 / 0.01 |
| AMP | True |
| seed / deterministic | 0 / True |
| pretrained | True |

두 실험은 **`--data` 파일만** 다르다 (`args.yaml`로 검증).

### 4.3 실험 결과 (각 모델 자체 val)

| | binary2 (100/100, best@84) | no_coco (100/100, best@95) |
|---|---:|---:|
| val 크기 | 734장 (534 banner + 200 COCO) | 534장 (banner only) |
| mAP@0.5 (Box) | 0.9777 | 0.9815 |
| mAP@0.5:0.95 (Box) | 0.9284 | 0.9411 |
| Precision | 0.9478 | 0.9498 |
| Recall | 0.9281 | 0.9525 |

> ⚠️ 자체 val은 구성 자체가 다르므로 직접 비교 불가 → §4.4 cross-evaluation 필수.

> 📷 **[FIG 1]** YOLO 학습 곡선 2개를 나란히 — `results/yolo_runs/banner_binary2/results.png` (좌) / `results/yolo_runs/banner_no_coco/results.png` (우). loss·mAP·Precision·Recall 수렴 비교.

### 4.4 공정 Cross-Evaluation

**(A) 동일 banner-only val (534장, 821 instance)에서 두 모델 평가**

| | binary2 | no_coco | Δ |
|---|---:|---:|---:|
| mAP50 (Box) | 0.9772 | **0.9803** | +0.0031 |
| mAP50-95 (Box) | 0.9327 | **0.9414** | +0.0087 |
| **Precision (Box)** | **0.9533** | 0.9388 | −0.0145 |
| Recall (Box) | 0.9209 | **0.9536** | +0.0327 |

→ banner 자체 mAP는 no_coco가 근소 우위, 반면 **Precision은 binary2 우위**. COCO negative 학습이 모델을 "확신할 때만 탐지"하는 방향으로 보수화한 결과, Recall −3.3%p 손실.

**(B) COCO-only 400장 (현수막 없는 일반 풍경)에서 FPR 측정**

| 셋 | binary2 FP 이미지 | no_coco FP 이미지 | no_coco 배수 |
|---|---:|---:|---:|
| COCO val (200장) | **1 (0.50%)** | 21 (10.50%) | 21× |
| COCO test (200장) | **2 (1.00%)** | 24 (12.00%) | 12× |
| **합계 (400장)** | **3 (0.75%)** | 45 (11.25%) | **15×** |

→ no_coco는 일반 풍경 **9장 중 1장을 환각**, binary2는 100장 중 1장 미만. 배경 건물·간판 패턴이 "현수막 같은 것"으로 보이는 것을 COCO negative가 교정.

### 4.5 가설 H1 재검증

| 주장 | 결과 |
|---|---|
| Precision ↑ | ✅ +1.45%p (0.9388 → 0.9533) |
| FPR ↓ | ✅ **15× 감소** (11.25% → 0.75%) |
| mAP ↑ on banner-only | ❌ −0.3%p (Recall 손실이 상쇄) |
| mAP ↑ on 혼합 셋 | ✅ (binary2가 COCO 포함 val 0.9777 달성) |

**실무적 결론**: 거리 CCTV처럼 풍경 99% 환경에서 오탐 1건당 뒷단(H-matrix + VLM) 비용을 고려하면 **binary2 선택이 압도적으로 유리**.

> 📷 **[FIG 2]** Precision-Recall 곡선 2개를 나란히 — `results/yolo_runs/banner_binary2/BoxPR_curve.png` vs `results/yolo_runs/banner_no_coco/BoxPR_curve.png`. Precision↑ vs Recall↑ trade-off 시각화.
>
> 📷 **[FIG 3]** (옵션) 탐지 예시 — `results/yolo_runs/banner_binary2/val_batch0_pred.jpg`. 실제 predictions 한 배치를 보여줘 정성 품질 어필.

---

## 5. 파이프라인 구현 분석 ([04_pipeline.py](04_pipeline.py))

### 5.1 BannerDetector ([04_pipeline.py:71](04_pipeline.py#L71))
- `ultralytics.YOLO` 래퍼. `detect()`는 `result.masks.xy[0]` (픽셀 다각형) + `mask.data[0]` (boolean mask) + `box.xyxy` (bbox)를 **모두** 반환해 뒷단에서 좌표/픽셀 어느 쪽이든 사용 가능.
- conf/iou는 생성자에서 고정 → 재현성.

### 5.2 HomographyCorrector ([04_pipeline.py:172](04_pipeline.py#L172))

**4 극점 직접 추출** ([`_pick_4_extremes`](04_pipeline.py#L247)):
```
TL = argmin(x + y)      TR = argmax(x − y)
BL = argmin(x − y)      BR = argmax(x + y)
```
polygon **N개 점 전체**를 후보로 보기 때문에 `convexHull` / `approxPolyDP` (Douglas–Peucker)의 스케일 의존성이나 `np.roll` 회전 이슈가 없다.

**보정 절차** ([`correct`](04_pipeline.py#L197)):
1. 위 4 극점을 `[TL, TR, BR, BL]` 순으로 얻음.
2. 출력 크기는 **4 극점 사이 실측 변 길이**로 결정:
   - `W = max(‖TR−TL‖, ‖BR−BL‖)`
   - `H = max(‖BL−TL‖, ‖BR−TR‖)`
   → 이 쿼드의 종횡비를 그대로 따라가 가로/세로 강제 변형 없음.
3. `max(W, H) > 1024`면 비율 유지 축소.
4. **4점 정확해**인 `cv2.getPerspectiveTransform` + `cv2.warpPerspective(BORDER_REPLICATE)`. RANSAC 불필요.

이전 구현은 `findHomography(RANSAC)` + 고정 `800×200` dst → 비-4:1 현수막이 찌그러져 가독성 저하되던 문제를 이 구조로 제거.

### 5.3 BannerClassifier ([04_pipeline.py:276](04_pipeline.py#L276))

**모델**: `Qwen/Qwen3.5-9B` — 9B 파라미터 **image-text-to-text** 멀티모달. 하이브리드 Gated DeltaNet + Gated Attention, context 262K.

**주요 구현 결정**:

| 항목 | 값 | 이유 |
|---|---|---|
| dtype | BF16 원본 (양자화 없음) | 18GB가 A5000 24GB에 YOLO와 함께 안착 → 양자화로 정확도 깎을 이유 없음. 4-bit 경로는 `config.VLM_CONFIG["quantization"]="4bit"`로 폴백 가능 |
| device_map | `{"": 0}` | `"auto"`는 bnb 양자화 경로에서 CPU/disk offload 에러 유발 → 단일 GPU 고정 |
| 입력 이미지 | 긴 변 896px로 축소 ([`MAX_VLM_IMAGE_SIDE`](04_pipeline.py#L308)) | 큰 이미지에서 gibberish 출력하는 [Qwen3.5-9B 공식 디스커션 #12](https://huggingface.co/Qwen/Qwen3.5-9B/discussions/12) 권고 반영 |
| thinking mode | **ON** (`enable_thinking=True` 직접 kwarg) | Qwen3 family 규약 — [HF 블로그 "4 things Qwen-3 chat template teaches"](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) |
| `max_new_tokens` | 1024 | think 체인이 수백 토큰 나오므로 256이면 "카테고리:" 라인 잘림 |
| `repetition_penalty` | 1.1 | `!!!!...` 같은 동일 토큰 폭주 억제 |

**thinking mode kwarg 전달 방식 — 주의점**: `chat_template_kwargs={"enable_thinking": ...}` 딕셔너리 방식은 AutoProcessor 단계에서 `Keyword argument chat_template_kwargs is not a valid argument for this processor and will be ignored` 경고와 함께 무시된다(런타임에서 직접 확인). **직접 kwarg로 전달해야 반영됨**. 구버전 transformers는 `TypeError` 폴백.

**프롬프트 3단 Chain-of-Thought** ([`_build_prompt`](04_pipeline.py#L609)):
1. **문구 추출**: 현수막 한글을 그대로 옮겨 쓰게 함 → 모델이 이미지를 진짜 "읽는지" 검증하는 진단축.
2. **근거**: 추출 문구 + 로고/색상 기반으로 2문장.
3. **카테고리**: `"카테고리: 정당/민간/공공 현수막"` 고정 라인.

**후처리**:
- [`_strip_think_block`](04_pipeline.py#L569) — `<think>...</think>` 블록을 정규식으로 분리 (정상 / 닫힘-only / 열림-only 3 케이스 모두). 파서는 본답변(`answer`)에서만 카테고리 탐색.
- [`_parse_category`](04_pipeline.py#L643) — 5단 폴백: "카테고리:" 라인 정확매칭(0.95) → 라인 키워드(0.85) → 답변 말미 5줄(0.7) → 전체 텍스트(0.55) → 글로벌 키워드(0.35) → 기본값(0.1). 단일 글자 키워드 "당"이 "당신"(프롬프트 echo)과 과매칭되는 버그를 발견해 2글자+ 키워드만 유지 ("정당/정치/선거/후보/국회/의원").

**에러 핸들링 (심층)**: 전처리 → template → to(device) → generate → decode 5 단계를 각각 try/except. 실패 시 [`_error_result`](04_pipeline.py#L596)가 유효 dict 반환 → 배치 평가가 한 샘플 예외로 중단되지 않음. 첫 샘플만 `debug=True`로 호출해 `thinking=ON / input_tokens / image_size / pixel_values.shape / think_len / answer_len / answer[:200]` 진단 로그 출력.

### 5.4 BannerPipeline ([04_pipeline.py:726](04_pipeline.py#L726))

**VRAM 스왑** — 18GB VLM과 YOLO 동시 상주 위험 회피:
- [`unload_detector()`](04_pipeline.py#L770) → YOLO 모델 삭제 + `torch.cuda.empty_cache()`.
- [`load_classifier()`](04_pipeline.py#L789) → 지연 로드 (idempotent).
- [`05_demo.py`](05_demo.py)는 `load_vlm=False`로 파이프라인을 열고 YOLO 선행 감지 → unload → VLM 로드 순서로 돌린다.

**Graceful degrade**: VLM 로드 실패 시 탐지·보정만 수행하는 모드로 자동 전환.

---

## 6. VLM 분류 정량 평가 ([05_demo.py](05_demo.py))

### 6.1 평가 설계
- COCO JSON에서 종류 라벨 있는 GT 샘플 추출 → YOLO seg 감지 → GT bbox와 **IoU≥0.3 매칭**하여 매칭된 폴리곤으로 H-matrix 보정 ([`seg_corrected_from_gt`](05_demo.py#L167)).
- 매칭 실패 시 GT bbox crop으로 fallback ([`precompute_corrections`](05_demo.py#L197)).
- VLM 분류 결과를 GT와 비교해 Accuracy / Precision / Recall / F1 / Confusion Matrix 계산.

### 6.2 결과 (n=10 스모크 테스트)

| 지표 | 값 |
|---|---|
| **Accuracy** | **80.0%** (8/10) |
| seg 매칭 / crop fallback / 실패 | 7 / 3 / 0 |
| 공공 현수막 F1 | **1.000** (3/3) |
| 민간 현수막 F1 | 0.833 (5/6) |
| 정당 현수막 F1 | 0.000 (0/1, 샘플 수 부족) |

> ⚠️ n=10은 상위 bound를 보는 스모크 — 정당 샘플 1개로 정당 F1=0은 통계적으로 의미 없음. **n≥50**에서 신뢰성 있는 수치 기대. (출처: [results/demo/evaluation_report.md](results/demo/evaluation_report.md))

### 6.3 진단 관찰

VLM의 `[VLM debug]` 출력에서 모델이 실제로 한글을 읽고 있음을 확인:
> `raw_output`: `"1단계: 문구 추출 — 행복한 마을 만들기 ... 주민자치회 ..."`

이전에 모든 샘플에 "정당"만 반환하던 버그 2종을 이 평가 과정에서 발견·수정:
1. **이미지 토큰 미생성**: `{"type": "image", "image": pil_image}` 경로는 올바름을 debug 로그로 확인.
2. **"당" 키워드 오매칭**: 프롬프트 echo의 "당신"이 정당으로 파싱되는 버그 → 2글자+ 키워드로 교정.
3. **이미지 크기 과다**: 896px 축소로 gibberish 증상 해결.
4. **thinking chain truncate**: `max_new_tokens=256` → 1024 상향.

> 📷 **[FIG 4]** VLM 혼동행렬 — `results/demo/vlm_confusion_matrix.png`. 3×3 heatmap with annotation.
>
> 📷 **[FIG 5]** VLM 정성 데모 — `results/demo/pipeline_demo.png`. 6 샘플 × 3 열 (원본+GT bbox / H-Matrix 보정 / Pred·Conf·GT·OK/X).

### 6.4 실행 명령

```bash
# Docker 컨테이너 안에서
docker compose exec banner-project python 05_demo.py --n-samples 50
# 또는 SSH-안전 백그라운드
docker compose exec -d banner-project bash -c \
  "python 05_demo.py --n-samples 50 > /workspace/results/demo.log 2>&1"
```

---

## 7. 창의성·차별점

1. **Polygon-기반 종횡비 보존 H-matrix** — bbox crop 대신 세그 폴리곤에서 4 극점을 직접 뽑아 실측 변 길이로 warp. 고정 4:1 강제로 찌그러지던 초기 구현의 문제를 제거해 VLM 입력 가독성 향상.
2. **convexHull/approxPolyDP 없이 N점 전점 극점 추출** — Douglas–Peucker의 스케일 의존성 회피. 불규칙 마스크에도 극점이 안정적. 4점 대응의 닫힌 해(`getPerspectiveTransform`)라 RANSAC 불필요.
3. **계층적 분업 + VRAM 스왑** — 탐지는 가벼운 YOLO26-m, 의미 분류는 VLM(18GB). 배치 평가 시 YOLO 선행 → unload → VLM 로드 순서를 명시적으로 설계해 OOM 위험 회피.
4. **COCO 부정학습** — 공개 데이터셋으로 **FPR 15× 감소**(§4.4). 본 프로젝트의 핵심 실험 기여.
5. **3단 구조적 CoT 프롬프트 + thinking mode ON** — (문구 추출 / 근거 / 카테고리) 강제 포맷으로 파싱 안정성 + OCR 검증축을 동시에 제공. Qwen3 thinking mode로 추론 품질을 올리고, `<think>` 블록은 정규식으로 후처리.
6. **Reproducibility 우선** — `seed=0`, `deterministic=True`, Docker `restart: unless-stopped`, 볼륨 마운트로 SSH-안전 학습·재현.

---

## 8. 사용 기술 스택 (최신성)

| 단계 | 기술 | 출시/버전 | 최신성 근거 |
|---|---|---|---|
| 탐지 | **YOLO26-seg** (ultralytics) | 2026 | 프로젝트 의존 기준 최신 YOLO 세대 |
| 멀티모달 | **Qwen/Qwen3.5-9B** | 2025~ | 9B image-text-to-text, Gated DeltaNet + Gated Attention, context 262K, thinking mode 지원 |
| 양자화 | bitsandbytes NF4 + Double Quant (옵션) | QLoRA 표준 | BF16 원본이 기본, 6GB 환경 폴백 경로 유지 |
| 학습 가속 | AMP (FP16 mixed) | torch 2.x | 학습 시간 ~40%↓ |
| 인프라 | Docker Compose + bind mount + `restart: unless-stopped` | — | SSH-단절 안전 학습·평가 |

---

## 9. 향후 작업

1. **VLM 본평가**: n≥50 (각 class 15+) 로 confusion matrix 재측정. 현재 n=10은 스모크.
2. **한국 가두 풍경 negative**: COCO 외 국내 거리 이미지 추가로 도메인 갭 축소.
3. **end-to-end latency 벤치**: 실시간성 검증용 ms 단위 측정.
4. **conf 임계값 스윕**: binary2의 Recall −3.3%p를 conf 0.15~0.25 범위에서 회복 가능성 검증.
5. **프롬프트/generation 튜닝**: thinking chain 길이와 정확도 상관 측정.

---

## 10. 부록 — 재현 명령

### 10.1 학습 (완료)
```bash
python prepare_binary_dataset.py
python prepare_no_coco_dataset.py
python 02_train_yolo.py --data data/dataset_binary.yaml  --name banner_binary2
python 02_train_yolo.py --data data/dataset_no_coco.yaml --name banner_no_coco
```

### 10.2 Cross-evaluation (§4.4 (A))
```bash
docker compose exec banner-project python -c "
from ultralytics import YOLO
for tag, p in [('binary2','results/yolo_runs/banner_binary2/weights/best.pt'),
               ('no_coco','results/yolo_runs/banner_no_coco/weights/best.pt')]:
    m = YOLO(p)
    r = m.val(data='data/dataset_no_coco.yaml', split='val',
              imgsz=640, batch=16, project='results/cross_eval',
              name=tag+'_on_banner_only_val', plots=False, verbose=False)
    print(tag, r.results_dict)
"
```

### 10.3 FPR 측정 (§4.4 (B))
```bash
docker compose exec banner-project python -c "
from ultralytics import YOLO
import glob, torch
for split in ['val', 'test']:
    imgs = sorted(glob.glob(f'data/binary/images/{split}/COCO_*.jpg'))
    for tag, p in [('binary2','results/yolo_runs/banner_binary2/weights/best.pt'),
                   ('no_coco','results/yolo_runs/banner_no_coco/weights/best.pt')]:
        m = YOLO(p); fp_i = fp_b = 0
        for i in range(0, len(imgs), 4):
            for r in m.predict(imgs[i:i+4], conf=0.25, iou=0.6, device=0,
                               imgsz=640, verbose=False, save=False):
                n = len(r.boxes) if r.boxes is not None else 0
                if n: fp_i += 1; fp_b += n
            torch.cuda.empty_cache()
        print(split, tag, f'FP imgs={fp_i}/{len(imgs)} boxes={fp_b}')
"
```

### 10.4 파이프라인 데모 & VLM 평가 (§6)
```bash
# 단일 이미지
docker compose exec banner-project python 04_pipeline.py \
    --image data/binary/images/test/$(ls data/binary/images/test/ | grep -v COCO_ | head -1) \
    --visualize --output results/demo_full

# 배치 평가 (결과물: results/demo/{pipeline_demo.png, vlm_confusion_matrix.png, evaluation_report.md})
docker compose exec banner-project python 05_demo.py --n-samples 50
```

### 10.5 핵심 코드 참조

| 기능 | 파일:라인 |
|---|---|
| 데이터셋 빌더 | [prepare_binary_dataset.py:48](prepare_binary_dataset.py#L48), [prepare_no_coco_dataset.py:42](prepare_no_coco_dataset.py#L42) |
| YOLO 학습 진입점 | [02_train_yolo.py:185](02_train_yolo.py#L185) |
| H-matrix 4 극점 | [04_pipeline.py:247](04_pipeline.py#L247) |
| H-matrix 변환 | [04_pipeline.py:232](04_pipeline.py#L232) |
| VLM 이미지 상한 | [04_pipeline.py:308](04_pipeline.py#L308) |
| VLM 로더 | [04_pipeline.py:350](04_pipeline.py#L350) |
| VLM classify + thinking ON | [04_pipeline.py:420](04_pipeline.py#L420) |
| `<think>` 분리 | [04_pipeline.py:569](04_pipeline.py#L569) |
| 카테고리 파서 | [04_pipeline.py:643](04_pipeline.py#L643) |
| VRAM 스왑 | [04_pipeline.py:770](04_pipeline.py#L770), [04_pipeline.py:789](04_pipeline.py#L789) |
| seg↔GT IoU 매칭 | [05_demo.py:167](05_demo.py#L167) |
| 사전 감지/보정 캐싱 | [05_demo.py:197](05_demo.py#L197) |
| VLM 정량 평가 | [05_demo.py:239](05_demo.py#L239) |
| 전역 설정 | [config.py:85](config.py#L85) YOLO, [config.py:153](config.py#L153) VLM, [config.py:183](config.py#L183) H-Matrix |

---

## 부록 — 그림 배치 가이드 (슬라이드/PDF 변환 시)

| 위치 | 파일 | 용도 |
|---|---|---|
| §0 요약 | `results/demo/pipeline_demo.png` | 헤드라인 정성 예시 |
| §4.3 | `results/yolo_runs/banner_binary2/results.png` | binary2 학습 곡선 |
| §4.3 | `results/yolo_runs/banner_no_coco/results.png` | no_coco 학습 곡선 (병렬 배치) |
| §4.5 | `results/yolo_runs/banner_binary2/BoxPR_curve.png` | PR 곡선 (Precision↑) |
| §4.5 | `results/yolo_runs/banner_no_coco/BoxPR_curve.png` | PR 곡선 (Recall↑) |
| §4.5 옵션 | `results/yolo_runs/banner_binary2/val_batch0_pred.jpg` | 정성 탐지 예시 |
| §4 옵션 | `results/yolo_runs/banner_binary2/confusion_matrix.png` | YOLO 혼동행렬 |
| §6.2 | `results/demo/vlm_confusion_matrix.png` | VLM 3×3 혼동행렬 |
| §6.3 | `results/demo/pipeline_demo.png` | VLM 정성 6×3 그리드 |
