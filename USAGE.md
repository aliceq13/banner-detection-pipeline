# 실행 가이드 (USAGE)

각 스크립트의 **실제 코드 진입점**(argparse, `main()`, import 의존성)을 분석해 정리했습니다. 주석이 아니라 코드 동작 기준으로 작성됐습니다.

---

## 1. 실행 환경

### 필수 조건
- **NVIDIA GPU + Docker + nvidia-container-toolkit** — `docker-compose.yml`이 `deploy.resources.reservations.devices`에 `nvidia` 요청함
- GPU 메모리 **최소 24GB** — Qwen3.5-9B BF16 로드 기준 (A5000 / A6000 / RTX 3090 / RTX 4090 등)

### 경로 모델
[`docker-compose.yml`](docker-compose.yml)의 볼륨은 **compose 파일 위치 기준 상대경로**입니다. 즉 저장소를 clone한 위치(`pwd`)가 그대로 컨테이너의 `/workspace/project`에 마운트됩니다. `config.py` 는 [`config.py:18-31`](config.py#L18-L31)에서 `/workspace/project` 존재 여부로 Docker/호스트 실행을 자동 구분합니다.

| 호스트 경로 | 컨테이너 경로 | 용도 | 필수 여부 |
|---|---|---|---|
| 저장소 루트 (`.`) | `/workspace/project` | 코드 | **필수** |
| `./.hf_cache/` | `/workspace/hf_cache` | Qwen·HF 모델 캐시 | **필수** (초기 공백이어도 무방) |
| `./results/` | `/workspace/results` | 학습 결과·모델·데모 산출물 | **필수** |
| 원천 `illegal_banner/` | `/workspace/illegal_banner` | GT 레이블·이미지 | **선택** (데이터 준비·`05_demo.py` 에만) |

원천 데이터셋을 쓰려면 `docker-compose.yml` 볼륨 목록 하단의 주석을 해제하고 실제 경로로 바꾸세요:
```yaml
# - /path/to/illegal_banner:/workspace/illegal_banner:ro
```

### 컨테이너 기동
```bash
git clone https://github.com/aliceq13/banner-detection-pipeline.git
cd banner-detection-pipeline
docker compose up -d                           # 이미지 빌드 + 컨테이너 기동 (bash 대기 상태)
docker compose exec banner-project bash        # 대화형 접속 (선택)
```

bind-mount 구조라 컨테이너 재기동 없이 호스트에서 코드를 수정해 바로 반영됩니다.

### 학습된 모델 다운로드 (학습 건너뛰기)

처음부터 학습하지 않고 바로 추론·평가를 돌리려면 학습된 YOLO 가중치를 내려받아 `results/models/yolo26_banner_best.pt` 위치에 두세요 ([`config.py:68`](config.py#L68)의 `YOLO_BEST_MODEL` 기본 경로).

```bash
mkdir -p results/models
curl -L -o results/models/yolo26_banner_best.pt \
    https://github.com/aliceq13/banner-detection-pipeline/releases/download/v1.0/binary2-best.pt
```

Ablation 비교용 no_coco 가중치:
```bash
mkdir -p results/yolo_runs/banner_no_coco/weights
curl -L -o results/yolo_runs/banner_no_coco/weights/best.pt \
    https://github.com/aliceq13/banner-detection-pipeline/releases/download/v1.0/no_coco-best.pt
```

---

## 2. 데이터 준비 (3-step)

### 2.1 `01_prepare_data.py` — COCO 다운로드 + 병합 + VLM 데이터 분할

인자 없음. `main()`이 순서대로 수행 (`01_prepare_data.py:540-578`):

> **⚠ 원천 데이터셋 필수.** 이 단계는 `/workspace/illegal_banner` 마운트(= 호스트의 원천 `illegal_banner/`)가 있어야 합니다. 데이터셋이 없다면 [§5 단일 이미지 파이프라인](#5-단일-이미지-파이프라인-04_pipelinepy)부터 바로 이용하세요.

1. `prepare_coco_negatives()` — COCO 2017 val + annotations zip 다운로드, `COCO_NEG_MAX_SAMPLES=2000` 장을 `data/coco_negative/`에 복사
2. `prepare_banner_positives()` — `DATA_ROOT/data/{images,labels}/{train,val,test}` 존재 여부 확인 ([`config.py:29`](config.py#L29)의 `DATA_ROOT`)
3. `create_merged_dataset()` — 현수막 positives + COCO negatives 를 `data/merged/`에 symlink + 빈 레이블 생성
4. `prepare_vlm_data()` — `temp_illegal_banner` COCO JSON 파싱해 `data/vlm_classification/{train,val}/{정당,민간,공공}/` 디렉토리로 이미지 분류 복사
5. `create_dataset_yaml()` — `data/dataset.yaml` (2-class: banner, text) 생성

```bash
docker compose exec banner-project python 01_prepare_data.py
```

> 최초 실행 시 COCO zip 약 1GB 다운로드. 이후 실행 시 `download_file()`이 존재 감지로 스킵합니다.

### 2.2 `prepare_binary_dataset.py` — banner 단일 클래스 이진 셋 (+ COCO neg)

인자 없음. `create_binary_dataset()`이 수행:

- `src/train,val,test`의 레이블 중 `startswith("0 ")` 라인만 남겨 **text 클래스(1) 제거**
- 이미지는 `symlink_to(img.resolve())` — 디스크 복사 없음
- `COCO_NEG_DIR/images/{split}` 의 COCO 이미지를 `COCO_` 접두사로 추가 + **빈 레이블 파일** 생성
- 산출물: `data/binary/`, `data/dataset_binary.yaml` (nc=1, names=[banner])

```bash
docker compose exec banner-project python prepare_binary_dataset.py
```

### 2.3 `prepare_no_coco_dataset.py` — Ablation용 banner-only 셋

`binary`와 동일하되 **COCO 부분만 생략**. 산출물: `data/no_coco/`, `data/dataset_no_coco.yaml`.

```bash
docker compose exec banner-project python prepare_no_coco_dataset.py
```

---

## 3. YOLO 학습 (`02_train_yolo.py`)

### 3.1 CLI 옵션 (`02_train_yolo.py:35-86`)

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--model` | `config.YOLO_CONFIG["model"]` = `yolo26m-seg.pt` | 초기 가중치 |
| `--epochs` | `config.YOLO_CONFIG["epochs"]` = 100 | 학습 에폭 |
| `--batch` | `config.YOLO_CONFIG["batch"]` = -1 (auto) | 배치 크기 |
| `--imgsz` | 640 | 입력 해상도 |
| `--device` | `0` | GPU 인덱스 |
| `--resume` | flag | 중단된 학습 이어가기 |
| `--data` | `config.DATASET_YAML` = `data/dataset.yaml` | 데이터셋 YAML |
| `--name` | `banner_detection` | 결과 하위 디렉토리명 |

학습 수행: `model.train(...)`에 AMP True, `val=True`(매 에폭 검증), `save=True`, early stopping `patience=50`. 완료 후 `best.pt`를 `config.YOLO_BEST_MODEL = results/models/yolo26_banner_best.pt`로 복사합니다.

### 3.2 본 프로젝트의 두 Ablation 학습

```bash
# (A) binary2 — COCO negative 포함 (권장)
docker compose exec banner-project python 02_train_yolo.py \
    --data data/dataset_binary.yaml \
    --name banner_binary2

# (B) no_coco — banner 전용, Ablation 비교군
docker compose exec banner-project python 02_train_yolo.py \
    --data data/dataset_no_coco.yaml \
    --name banner_no_coco
```

결과 저장 위치: `results/yolo_runs/{name}/` (weights, results.png, PR curves, confusion_matrix.png 등).

**SSH 단절 안전 백그라운드 실행**:
```bash
docker compose exec -d banner-project bash -c \
  "python 02_train_yolo.py --data data/dataset_binary.yaml --name banner_binary2 \
   > /workspace/results/log_binary_train.txt 2>&1"
```

---

## 4. YOLO 평가 (`03_evaluate_yolo.py`)

### CLI 옵션 (`03_evaluate_yolo.py:37-75`)

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--model` | `config.YOLO_BEST_MODEL` | 평가할 `.pt` 경로 |
| `--data` | `config.DATASET_YAML` | 데이터셋 YAML |
| `--split` | `test` | `train`/`val`/`test` |
| `--conf` | 0.25 | confidence threshold |
| `--iou` | 0.6 | NMS IoU |
| `--save-samples` | 20 | 저장할 시각화 샘플 수 |

```bash
docker compose exec banner-project python 03_evaluate_yolo.py \
    --model results/yolo_runs/banner_binary2/weights/best.pt \
    --data data/dataset_binary.yaml \
    --split test
```

mAP50, mAP50-95, Precision, Recall을 콘솔에 찍고 샘플 이미지를 저장합니다.

---

## 5. 단일 이미지 파이프라인 (`04_pipeline.py`)

### CLI 옵션 (`04_pipeline.py:1006-1046`)

| 플래그 | 설명 |
|---|---|
| `--image <path>` | 단일 이미지 경로 (필수, 혹은 `--batch`) |
| `--batch <dir>` | 디렉토리 내 모든 `.jpg`/`.png` 일괄 처리 |
| `--model <path>` | YOLO 가중치 (기본 `config.YOLO_BEST_MODEL`) |
| `--output <dir>` | 결과 저장 디렉토리 (기본 `results/pipeline_output`) |
| `--visualize` | 단계별 시각화 이미지 저장 |
| `--no-vlm` | VLM 로드 생략 (탐지+보정까지만) |
| `--conf` | YOLO confidence threshold |

`main()` 흐름 (`04_pipeline.py:1049-1104`):
1. `BannerPipeline(yolo_model_path, load_vlm=True)` 생성 — 생성 시 YOLO + VLM(Qwen3.5-9B) **동시 로드**
2. `pipeline.process(image_path, visualize, save_dir)` 실행
3. 각 현수막별로 `{category, confidence, reason}` 출력

```bash
# 본인의 현수막 사진을 저장소 루트에 sample.jpg 로 저장한 뒤:
docker compose exec banner-project python 04_pipeline.py \
    --image sample.jpg --visualize --output results/pipeline_output

# 디렉토리 전체를 일괄 처리:
docker compose exec banner-project python 04_pipeline.py \
    --batch my_banners/ --visualize
```

- [`04_pipeline.py:101-105`](04_pipeline.py#L101-L105)에서 모델 파일이 없으면 `FileNotFoundError` + 안내 메시지를 띄웁니다.
- [`04_pipeline.py:826-828`](04_pipeline.py#L826-L828)에서 `cv2.imread`가 실패하면 `ValueError("이미지 로드 실패: ...")` 발생.
- 최초 실행 시 Qwen3.5-9B(~18GB)가 `.hf_cache/`로 다운로드됩니다. 이후 실행은 캐시에서 로드.

> 단일 이미지용. 배치 평가에는 VRAM 효율적인 `05_demo.py`를 사용하세요.

---

## 6. 배치 데모 & VLM 평가 (`05_demo.py`) — 핵심 평가 스크립트

> **⚠ 원천 데이터셋 필수.** [`05_demo.py:88-89`](05_demo.py#L88-L89)의 `load_gt_labels`가 `config.COCO_BANNER_LBL` (`/workspace/illegal_banner/temp_illegal_banner/illegal_banner/label/*.json`) 을 읽습니다. 데이터가 없으면 "No GT samples loaded. Check temp data." 출력 후 `sys.exit`. 데이터셋을 `docker-compose.yml`에서 마운트하세요.

### CLI 옵션 (`05_demo.py:54-65`)

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--n-samples` | 50 | 평가할 GT 샘플 수 |
| `--skip-vlm` | flag | VLM 평가 건너뜀 (빠른 확인) |
| `--model` | `config.YOLO_BEST_MODEL` | YOLO 가중치 |
| `--output` | `results/demo` | 결과 디렉토리 |
| `--seed` | 42 | 샘플 선택 시드 |

### `main()` 흐름 (`05_demo.py:508-578`)

코드 기반 실제 흐름:
1. `BannerPipeline(load_vlm=False)` — **VLM 먼저 로드하지 않음**
2. `load_gt_labels(n_samples, seed)` — COCO JSON에서 카테고리 있는 샘플 무작위 추출
3. `precompute_corrections(pipeline, gt_samples)` — 모든 샘플에 대해 YOLO 감지 + GT bbox 매칭(IoU≥0.3) + H-matrix 보정 **캐싱**
4. `pipeline.unload_detector()` → `pipeline.load_classifier()` — **VRAM 스왑** (YOLO 내리고 VLM 올리기)
5. `evaluate_vlm_classification(cached)` — 캐시된 보정 이미지에 VLM 적용, Accuracy/Precision/Recall/F1/Confusion Matrix 계산
6. `create_presentation_figure(cached, n_display=6)` — 6행×3열 정성 그리드 생성
7. `_write_summary_report()` — `evaluation_report.md` 기록

### 산출물 (`results/demo/`)
- `pipeline_demo.png` — 정성 비교 그림 (6 샘플 × 3 열)
- `vlm_confusion_matrix.png` — 3-class 혼동행렬
- `evaluation_report.md` — 정량 요약 (Accuracy, 클래스별 P/R/F1)

### 실행
```bash
# 정식 평가 (n=50)
docker compose exec banner-project python 05_demo.py --n-samples 50

# 빠른 확인 (VLM 생략, YOLO+보정까지만)
docker compose exec banner-project python 05_demo.py --n-samples 10 --skip-vlm

# SSH 단절 안전 백그라운드
docker compose exec -d banner-project bash -c \
  "python 05_demo.py --n-samples 50 > /workspace/results/log_demo.txt 2>&1"
```

---

## 7. 전체 파이프라인 재현 (From Scratch)

```bash
# 0. 컨테이너 기동
docker compose up -d

# 1. 데이터 준비
docker compose exec banner-project python 01_prepare_data.py
docker compose exec banner-project python prepare_binary_dataset.py
docker compose exec banner-project python prepare_no_coco_dataset.py

# 2. YOLO 학습 (시간 소요: A5000 기준 ~6-10시간/모델)
docker compose exec -d banner-project bash -c \
  "python 02_train_yolo.py --data data/dataset_binary.yaml --name banner_binary2 \
   > /workspace/results/log_binary_train.txt 2>&1"

docker compose exec -d banner-project bash -c \
  "python 02_train_yolo.py --data data/dataset_no_coco.yaml --name banner_no_coco \
   > /workspace/results/log_no_coco_train.txt 2>&1"

# 3. 학습 진행 모니터링
docker compose exec banner-project tail -f /workspace/results/log_binary_train.txt

# 4. YOLO 평가 (학습 완료 후)
docker compose exec banner-project python 03_evaluate_yolo.py \
    --model results/yolo_runs/banner_binary2/weights/best.pt \
    --data data/dataset_binary.yaml --split test

# 5. 파이프라인 데모 + VLM 평가
docker compose exec banner-project python 05_demo.py --n-samples 50
```

---

## 8. 설정 변경 (`config.py`)

런타임 옵션은 거의 모두 `config.py`에 중앙화돼 있습니다.

- `YOLO_CONFIG`: 모델·에폭·배치·lr·patience (`config.py:85-120`)
- `VLM_CONFIG`: model_id, quantization(None/"8bit"/"4bit"), max_new_tokens(1024), temperature, cache_dir (`config.py:153-180`)
- `HMATRIX_CONFIG`: max_size 1024 (`config.py:186-191`)
- `COCO_NEG_MAX_SAMPLES = 2000` (`config.py:79`) — COCO 샘플 수 상한

`config.py:18`의 `_IS_DOCKER = os.path.exists("/workspace/project")`로 Docker vs 호스트 환경을 **자동 감지** 후 경로를 분기합니다.

---

## 9. 호스트 직접 실행 (Docker 없이)

[`config.py:26-31`](config.py#L26-L31)에서 호스트 경로가 정의돼 있지만, 다음이 전제입니다:

- Python 3.10+, CUDA 12.1 호환 PyTorch 2.x
- `pip install -r requirements.txt` (bitsandbytes·transformers 등 ~10GB 설치)
- `01_prepare_data.py` / `05_demo.py` 를 쓰려면 호스트 `/data/illegal_banner` (기본값, [`config.py:29`](config.py#L29))에 원천 데이터셋이 존재해야 함 — 경로를 바꾸려면 `config.py` 편집
- 환경변수는 필수 아님 — `config.py`가 BASE_DIR 기준으로 `.hf_cache/`, `results/`를 자동 해석

```bash
pip install -r requirements.txt
python 04_pipeline.py --image sample.jpg --visualize   # 추론만 (데이터셋 불필요)
python 01_prepare_data.py                              # 데이터셋 필요
```

> 프로젝트는 **Docker 실행을 전제로 설계**돼 있으므로, 호스트 직접 실행은 CUDA/드라이버 버전 매칭 이슈가 발생할 수 있습니다.

---

## 10. 트러블슈팅 체크리스트 (코드 기준)

| 증상 | 원인 | 해결 |
|---|---|---|
| `YOLO 가중치가 없습니다: results/models/yolo26_banner_best.pt` | 가중치 미다운로드 | §1의 curl 명령 실행 |
| `이미지 파일을 찾을 수 없습니다: sample.jpg` | 파일 경로 오류 | 저장소 루트 기준 상대경로 확인 |
| `GT 레이블 디렉토리가 없습니다` (05_demo.py) | `illegal_banner` 마운트 누락 | docker-compose.yml 볼륨 주석 해제 또는 04_pipeline.py 사용 |
| VLM OOM | GPU 24GB 미만 | `VLM_CONFIG["quantization"]="4bit"` ([`config.py:164`](config.py#L164)) |
| VLM 응답이 잘림 | `max_new_tokens` 부족 (thinking 예산) | [`config.py:173`](config.py#L173) 조정 |
| VLM 이미지 입력 gibberish | 이미지가 너무 큼 | `MAX_VLM_IMAGE_SIDE=896` ([`04_pipeline.py`](04_pipeline.py)) |
| `VRAM 경합` (YOLO + VLM 동시 상주) | YOLO + VLM 동시 로드 | `load_vlm=False` 후 `unload_detector→load_classifier` 스왑 사용 |
| Qwen 최초 다운로드 실패 | 네트워크/rate limit | `HF_TOKEN` 환경변수 설정 (docker-compose.yml 주석 참조) |

---

## 빠른 참조 — 스크립트 한 줄 요약

| 파일 | 역할 | 주요 인자 |
|---|---|---|
| `01_prepare_data.py` | COCO 다운 + 병합 + VLM 분할 | (없음) |
| `prepare_binary_dataset.py` | banner 단일 클래스 + COCO neg | (없음) |
| `prepare_no_coco_dataset.py` | banner only (ablation 비교군) | (없음) |
| `02_train_yolo.py` | YOLO26-seg 학습 | `--data --name --epochs` |
| `03_evaluate_yolo.py` | YOLO val/test 평가 | `--model --data --split` |
| `04_pipeline.py` | 단일/배치 이미지 E2E | `--image \| --batch --visualize` |
| `05_demo.py` | GT 매칭 기반 VLM 정량 평가 + 정성 그리드 | `--n-samples --skip-vlm` |
