"""
prepare_binary_dataset.py - banner/non-banner 이진 분류 데이터셋 생성
=====================================================================
기존 데이터에서 text(class 1) 제거, banner(class 0)만 유지
COCO 이미지는 빈 레이블로 negative sample 역할
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config

BINARY_DIR = config.DATA_DIR / "binary"

def create_binary_dataset():
    print("=" * 50)
    print("이진 데이터셋 생성 (banner vs non-banner)")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        dst_img = BINARY_DIR / "images" / split
        dst_lbl = BINARY_DIR / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_pos, n_neg = 0, 0

        # ── Positive: banner 레이블만 유지 (text 제거) ──────────
        src_img_dir = config.YOLO_IMAGES / split
        src_lbl_dir = config.YOLO_LABELS / split

        if src_img_dir.exists():
            for img in tqdm(list(src_img_dir.glob("*.jpg")), desc=f"[{split}] banner"):
                lbl = src_lbl_dir / (img.stem + ".txt")

                # 이미지 심볼릭 링크
                dst = dst_img / img.name
                if not dst.exists():
                    dst.symlink_to(img.resolve())

                # 레이블: class 0(banner)만 유지, class 1(text) 제거
                dst_lbl_file = dst_lbl / (img.stem + ".txt")
                if lbl.exists():
                    lines = lbl.read_text().splitlines()
                    banner_lines = [l for l in lines if l.startswith("0 ")]
                    dst_lbl_file.write_text("\n".join(banner_lines))
                else:
                    dst_lbl_file.write_text("")
                n_pos += 1

        # ── Negative: COCO 이미지 + 빈 레이블 ───────────────────
        neg_img_dir = config.COCO_NEG_DIR / "images" / split
        if neg_img_dir.exists():
            for img in tqdm(list(neg_img_dir.glob("*.jpg")), desc=f"[{split}] COCO neg"):
                dst = dst_img / ("COCO_" + img.name)
                if not dst.exists():
                    dst.symlink_to(img.resolve())
                # 빈 레이블 = 현수막 없음
                (dst_lbl / ("COCO_" + img.stem + ".txt")).write_text("")
                n_neg += 1

        print(f"  [{split}] positive={n_pos}, negative={n_neg}, total={n_pos+n_neg}")

    # ── dataset_binary.yaml 생성 ─────────────────────────────────
    yaml_path = config.DATA_DIR / "dataset_binary.yaml"
    yaml_path.write_text(f"""path: {BINARY_DIR}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: banner
""")
    print(f"\n[DONE] {yaml_path}")
    print("다음: python 02_train_yolo.py --data data/dataset_binary.yaml")

if __name__ == "__main__":
    create_binary_dataset()
