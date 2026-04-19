"""
01_prepare_data.py - 학습 데이터 준비 스크립트
================================================
이 스크립트가 하는 일:
  1. 기존 현수막 데이터 확인 (YOLO segmentation 포맷)
  2. COCO 2017 val 세트 다운로드 (현수막 아닌 이미지 = negative sample)
  3. COCO negative + 현수막 positive를 병합한 학습 데이터셋 생성
  4. VLM 분류용 데이터 준비 (temp JSON → 클래스별 이미지 복사)
  5. dataset.yaml 파일 생성

왜 COCO negative가 필요한가?
  - 현수막 데이터만으로 학습하면 모델이 '현수막 아닌 것'을 잘 모름
  - COCO는 일상 환경의 다양한 객체(사람, 차, 건물 등)를 포함 → 강건한 negative
  - False Positive(현수막이 아닌 것을 현수막으로 탐지)를 줄임

데이터셋 구성:
  - Positive: 현수막 있는 이미지 (기존 YOLO 데이터)
  - Negative: 현수막 없는 COCO 이미지 (빈 레이블 파일)
"""

import os
import sys
import json
import shutil
import zipfile
import random
import requests
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가 (config.py import용)
sys.path.insert(0, str(Path(__file__).parent))
import config


# ============================================================
# 유틸리티 함수
# ============================================================

def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """
    URL에서 파일을 다운로드합니다.

    Args:
        url: 다운로드할 파일의 URL
        dest_path: 저장할 경로
        desc: 진행바에 표시할 설명

    Returns:
        True: 성공, False: 실패
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미 존재하면 건너뜀
    if dest_path.exists():
        print(f"  [SKIP] 이미 존재함: {dest_path.name}")
        return True

    try:
        print(f"  [DOWN] {desc}: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        print(f"  [DONE] 다운로드 완료: {dest_path}")
        return True

    except requests.RequestException as e:
        print(f"  [ERROR] 다운로드 실패: {e}")
        if dest_path.exists():
            dest_path.unlink()  # 불완전한 파일 삭제
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    ZIP 파일을 압축 해제합니다.

    Args:
        zip_path: ZIP 파일 경로
        extract_to: 압축 해제할 디렉토리
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  [UNZIP] {zip_path.name} → {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print(f"  [DONE] 압축 해제 완료")
        return True
    except zipfile.BadZipFile as e:
        print(f"  [ERROR] ZIP 파일 손상: {e}")
        return False


def copy_with_symlink(src: Path, dst: Path):
    """
    파일을 복사합니다. 디스크 공간 절약을 위해 심볼릭 링크 시도 후 복사.

    Args:
        src: 원본 파일 경로
        dst: 대상 파일 경로
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            # 심볼릭 링크 시도 (디스크 공간 절약)
            dst.symlink_to(src.resolve())
        except (OSError, NotImplementedError):
            # 실패하면 일반 복사
            shutil.copy2(src, dst)


# ============================================================
# Step 1: COCO 데이터 다운로드 및 준비
# ============================================================

def prepare_coco_negatives(
    max_samples: int = config.COCO_NEG_MAX_SAMPLES,
    seed: int = 42
) -> int:
    """
    COCO 2017 val 이미지를 negative sample로 준비합니다.

    전략:
      - COCO val 5000장에서 max_samples개 랜덤 샘플링
      - 각 이미지에 대해 빈 레이블 파일 생성 (= 현수막 없음)
      - 기존 학습/검증/테스트 비율(8:1:1)로 분할

    Args:
        max_samples: 사용할 최대 COCO 이미지 수
        seed: 재현성을 위한 랜덤 시드

    Returns:
        준비된 이미지 수
    """
    print("\n" + "="*60)
    print("Step 1: COCO Negative Sample 준비")
    print("="*60)

    # 다운로드 디렉토리
    coco_download_dir = config.DATA_DIR / "coco_download"
    coco_download_dir.mkdir(parents=True, exist_ok=True)

    # ── 1-1. COCO val 이미지 다운로드 ────────────────────────
    coco_zip = coco_download_dir / "val2017.zip"
    coco_img_dir = coco_download_dir / "val2017"

    if not coco_img_dir.exists():
        success = download_file(
            config.COCO_VAL_URL,
            coco_zip,
            desc="COCO val2017"
        )
        if not success:
            print("[WARNING] COCO 다운로드 실패. COCO negative 없이 진행합니다.")
            return 0

        extract_zip(coco_zip, coco_download_dir)
        # 용량 절약을 위해 ZIP 삭제
        if coco_zip.exists():
            coco_zip.unlink()
    else:
        print(f"  [SKIP] COCO val2017 이미 존재: {coco_img_dir}")

    # ── 1-2. COCO 이미지 목록 수집 ───────────────────────────
    all_coco_images = list(coco_img_dir.glob("*.jpg"))
    print(f"  [INFO] COCO val 이미지 총 {len(all_coco_images)}장")

    if len(all_coco_images) == 0:
        print("[ERROR] COCO 이미지를 찾을 수 없습니다.")
        return 0

    # ── 1-3. 랜덤 샘플링 ─────────────────────────────────────
    random.seed(seed)
    sampled = random.sample(all_coco_images, min(max_samples, len(all_coco_images)))
    print(f"  [INFO] {len(sampled)}장 샘플링 (최대 {max_samples}장)")

    # ── 1-4. train/val/test 분할 (8:1:1) ─────────────────────
    n = len(sampled)
    n_val  = n // 10
    n_test = n // 10
    n_train = n - n_val - n_test

    splits = {
        "train": sampled[:n_train],
        "val":   sampled[n_train:n_train+n_val],
        "test":  sampled[n_train+n_val:],
    }
    print(f"  [INFO] 분할: train={n_train}, val={n_val}, test={n_test}")

    # ── 1-5. 이미지 복사 + 빈 레이블 생성 ────────────────────
    for split, images in splits.items():
        img_dst_dir = config.COCO_NEG_DIR / "images" / split
        lbl_dst_dir = config.COCO_NEG_DIR / "labels" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(images, desc=f"COCO {split}"):
            # 이미지 복사 (심볼릭 링크 시도)
            dst_img = img_dst_dir / img_path.name
            copy_with_symlink(img_path, dst_img)

            # 빈 레이블 파일 생성 (현수막이 없음을 의미)
            # YOLO 포맷: 빈 파일 = 배경 이미지 (아무 객체 없음)
            lbl_file = lbl_dst_dir / (img_path.stem + ".txt")
            if not lbl_file.exists():
                lbl_file.write_text("")  # 빈 파일

    print(f"  [DONE] COCO negative 준비 완료: {n}장")
    return n


# ============================================================
# Step 2: 기존 YOLO 데이터 확인 및 복사
# ============================================================

def prepare_banner_positives() -> dict:
    """
    기존 현수막 YOLO 세그멘테이션 데이터를 확인하고 통계를 반환합니다.

    기존 데이터 포맷:
      - 이미지: /data/illegal_banner/data/images/{train,val,test}/*.jpg
      - 레이블: /data/illegal_banner/data/labels/{train,val,test}/*.txt
      - 레이블 형식: class x1 y1 x2 y2 x3 y3 x4 y4 (정규화 좌표)
        - class 0: banner (현수막 polygon)
        - class 1: text (현수막 내 텍스트)

    Returns:
        각 split별 이미지 수 딕셔너리
    """
    print("\n" + "="*60)
    print("Step 2: 기존 현수막 데이터 확인")
    print("="*60)

    counts = {}
    for split in ["train", "val", "test"]:
        img_dir = config.YOLO_IMAGES / split
        lbl_dir = config.YOLO_LABELS / split

        n_img = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        counts[split] = n_img

        print(f"  [{split:5s}] 이미지: {n_img:5d}장, 레이블: {n_lbl:5d}개")

    return counts


# ============================================================
# Step 3: 병합 데이터셋 생성 (Positive + Negative)
# ============================================================

def create_merged_dataset() -> dict:
    """
    현수막 positive 데이터와 COCO negative 데이터를 병합하여
    최종 학습 데이터셋을 생성합니다.

    심볼릭 링크를 사용하여 디스크 공간을 절약합니다.

    Returns:
        각 split별 최종 이미지 수
    """
    print("\n" + "="*60)
    print("Step 3: 병합 데이터셋 생성")
    print("="*60)

    counts = {}
    for split in ["train", "val", "test"]:
        dst_img = config.MERGED_DIR / "images" / split
        dst_lbl = config.MERGED_DIR / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_added = 0

        # ── Positive: 현수막 데이터 ────────────────────────────
        pos_img_dir = config.YOLO_IMAGES / split
        pos_lbl_dir = config.YOLO_LABELS / split

        if pos_img_dir.exists():
            for img_file in tqdm(
                list(pos_img_dir.glob("*.jpg")),
                desc=f"[{split}] Positive"
            ):
                lbl_file = pos_lbl_dir / (img_file.stem + ".txt")

                # 이미지 심볼릭 링크
                copy_with_symlink(img_file, dst_img / img_file.name)

                # 레이블 심볼릭 링크 (없으면 빈 파일)
                if lbl_file.exists():
                    copy_with_symlink(lbl_file, dst_lbl / lbl_file.name)
                else:
                    (dst_lbl / (img_file.stem + ".txt")).write_text("")

                n_added += 1

        # ── Negative: COCO 데이터 ──────────────────────────────
        neg_img_dir = config.COCO_NEG_DIR / "images" / split
        neg_lbl_dir = config.COCO_NEG_DIR / "labels" / split

        if neg_img_dir.exists():
            for img_file in tqdm(
                list(neg_img_dir.glob("*.jpg")),
                desc=f"[{split}] Negative"
            ):
                lbl_file = neg_lbl_dir / (img_file.stem + ".txt")

                # 파일명 충돌 방지: COCO_ 접두사 추가
                new_name = "COCO_" + img_file.name
                copy_with_symlink(img_file, dst_img / new_name)

                # 빈 레이블 파일
                (dst_lbl / ("COCO_" + img_file.stem + ".txt")).write_text("")

                n_added += 1

        counts[split] = n_added
        print(f"  [{split:5s}] 총 {n_added}장 병합")

    return counts


# ============================================================
# Step 4: VLM 분류용 데이터 준비
# ============================================================

def prepare_vlm_data(train_ratio: float = 0.8, seed: int = 42) -> dict:
    """
    VLM(Qwen3.5-9B 등) 분류를 위한 데이터를 준비합니다.

    temp 데이터의 COCO JSON 파일에서:
      - 현수막 종류(정당/민간/공공)별로 이미지를 정리
      - 각 이미지에서 현수막 영역을 crop하여 저장

    데이터 구조:
      vlm_classification/
        train/
          정당현수막/   ← crop된 현수막 이미지
          민간현수막/
          공공현수막/
        val/
          ...

    Args:
        train_ratio: 학습 데이터 비율 (나머지는 val)
        seed: 랜덤 시드

    Returns:
        각 클래스별 이미지 수
    """
    print("\n" + "="*60)
    print("Step 4: VLM 분류용 데이터 준비")
    print("="*60)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  [ERROR] OpenCV 미설치. pip install opencv-python")
        return {}

    # ── 4-1. COCO JSON 파싱 ──────────────────────────────────
    label_dir  = config.COCO_BANNER_LBL
    image_dir  = config.COCO_BANNER_IMG

    class_names = {
        "정당 현수막": "정당현수막",
        "민간 현수막": "민간현수막",
        "공공 현수막": "공공현수막",
    }

    # 클래스별 (이미지경로, bbox, segmentation) 수집
    samples = {cls: [] for cls in class_names.values()}

    json_files = list(label_dir.glob("*.json"))
    print(f"  [INFO] JSON 파일 수: {len(json_files)}")

    for json_file in tqdm(json_files, desc="JSON 파싱"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [WARN] JSON 파싱 실패: {json_file.name}: {e}")
            continue

        # 카테고리 ID → 이름 매핑
        cat_map = {c['id']: c['name'] for c in data.get('categories', [])}

        # 이미지 ID → 파일명 매핑
        img_map = {img['id']: img['file_name'] for img in data.get('images', [])}

        for ann in data.get('annotations', []):
            cat_name = cat_map.get(ann['category_id'], '')
            if cat_name not in class_names:
                continue  # text 등 불필요한 카테고리 건너뜀

            img_file = img_map.get(ann['image_id'], '')
            img_path = image_dir / img_file
            if not img_path.exists():
                continue

            # bounding box: [x, y, width, height]
            bbox = ann.get('bbox', [])
            if len(bbox) < 4 or bbox[2] < 10 or bbox[3] < 10:
                continue  # 너무 작은 영역 건너뜀

            samples[class_names[cat_name]].append({
                'img_path': img_path,
                'bbox': bbox,
                'ann_id': ann['id'],
            })

    # ── 4-2. 통계 출력 ──────────────────────────────────────
    for cls, sample_list in samples.items():
        print(f"  {cls}: {len(sample_list)}개")

    # ── 4-3. 이미지 crop 및 저장 ────────────────────────────
    random.seed(seed)
    class_counts = {}

    for cls_name, sample_list in samples.items():
        if not sample_list:
            continue

        # train/val 분할
        random.shuffle(sample_list)
        n_train = int(len(sample_list) * train_ratio)
        splits = {
            "train": sample_list[:n_train],
            "val":   sample_list[n_train:],
        }

        class_counts[cls_name] = len(sample_list)

        for split, split_samples in splits.items():
            dst_dir = config.VLM_DATA_DIR / split / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for item in tqdm(split_samples, desc=f"Crop [{cls_name}/{split}]"):
                img = cv2.imread(str(item['img_path']))
                if img is None:
                    continue

                x, y, w, h = [int(v) for v in item['bbox']]
                # 이미지 경계 클리핑
                ih, iw = img.shape[:2]
                x  = max(0, x)
                y  = max(0, y)
                x2 = min(iw, x + w)
                y2 = min(ih, y + h)

                crop = img[y:y2, x:x2]
                if crop.size == 0:
                    continue

                out_path = dst_dir / f"{item['ann_id']}.jpg"
                cv2.imwrite(str(out_path), crop)

    print(f"  [DONE] VLM 데이터 준비 완료")
    return class_counts


# ============================================================
# Step 5: YOLO dataset.yaml 생성
# ============================================================

def create_dataset_yaml(merged_counts: dict):
    """
    YOLO 학습에 필요한 dataset.yaml 파일을 생성합니다.

    YOLO는 이 파일을 통해 데이터 경로와 클래스 정보를 가져옵니다.

    Args:
        merged_counts: create_merged_dataset()에서 반환된 split별 이미지 수
    """
    print("\n" + "="*60)
    print("Step 5: dataset.yaml 생성")
    print("="*60)

    # Docker 컨테이너 내부에서 바라보는 데이터 경로
    if config._IS_DOCKER:
        merged_path = "/workspace/project/data/merged"
    else:
        merged_path = str(config.MERGED_DIR)

    yaml_content = f"""# ============================================================
# YOLO26 학습 데이터셋 설정
# ============================================================
# 자동 생성됨: {__file__}
# ============================================================

# 데이터셋 루트 디렉토리
path: {merged_path}

# 학습/검증/테스트 이미지 경로 (path 기준 상대경로)
train: images/train
val:   images/val
test:  images/test

# 클래스 수
nc: 2

# 클래스 이름 목록
# 0: banner - 현수막 영역 (세그멘테이션 대상)
# 1: text   - 현수막 내 텍스트 영역
names:
  0: banner
  1: text

# 데이터셋 통계 (참고용)
# train: {merged_counts.get('train', 0)}장 (positive + COCO negative 포함)
# val:   {merged_counts.get('val', 0)}장
# test:  {merged_counts.get('test', 0)}장
"""

    config.DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    config.DATASET_YAML.write_text(yaml_content, encoding='utf-8')
    print(f"  [DONE] 저장됨: {config.DATASET_YAML}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """
    데이터 준비 전체 파이프라인을 실행합니다.
    """
    print("=" * 60)
    print("현수막 탐지 프로젝트 - 데이터 준비")
    print("=" * 60)

    # 필요한 디렉토리 생성
    config.ensure_dirs()

    # Step 1: COCO negative 준비 (인터넷 연결 필요)
    coco_count = prepare_coco_negatives()

    # Step 2: 기존 현수막 데이터 확인
    banner_counts = prepare_banner_positives()

    # Step 3: 병합 데이터셋 생성
    merged_counts = create_merged_dataset()

    # Step 4: VLM 분류용 데이터 준비
    vlm_counts = prepare_vlm_data()

    # Step 5: dataset.yaml 생성
    create_dataset_yaml(merged_counts)

    # ── 최종 요약 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("데이터 준비 완료 - 최종 요약")
    print("=" * 60)
    print(f"현수막 데이터:     {sum(banner_counts.values()):6d}장")
    print(f"COCO negative:     {coco_count:6d}장")
    print(f"병합 데이터셋:")
    for split, n in merged_counts.items():
        print(f"  {split:5s}: {n:6d}장")
    print(f"VLM 분류 데이터:")
    for cls, n in vlm_counts.items():
        print(f"  {cls}: {n:6d}장")
    print(f"\n다음 단계: python 02_train_yolo.py")


if __name__ == "__main__":
    main()
