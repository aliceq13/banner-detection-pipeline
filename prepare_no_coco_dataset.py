"""
prepare_no_coco_dataset.py - COCO negative 없이 banner만으로 데이터셋 생성
비교 실험용: banner_binary2(COCO 있음) vs banner_no_coco(COCO 없음)
"""

import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

NO_COCO_DIR = config.DATA_DIR / "no_coco"

def create_no_coco_dataset():
    print("=" * 50)
    print("COCO negative 없는 데이터셋 생성")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        dst_img = NO_COCO_DIR / "images" / split
        dst_lbl = NO_COCO_DIR / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        src_img_dir = config.YOLO_IMAGES / split
        src_lbl_dir = config.YOLO_LABELS / split

        n = 0
        if src_img_dir.exists():
            for img in tqdm(list(src_img_dir.glob("*.jpg")), desc=f"[{split}]"):
                lbl = src_lbl_dir / (img.stem + ".txt")

                dst = dst_img / img.name
                if not dst.exists():
                    dst.symlink_to(img.resolve())

                # class 0(banner)만 유지, class 1(text) 제거
                dst_lbl_file = dst_lbl / (img.stem + ".txt")
                if lbl.exists():
                    lines = lbl.read_text().splitlines()
                    banner_lines = [l for l in lines if l.startswith("0 ")]
                    dst_lbl_file.write_text("\n".join(banner_lines))
                else:
                    dst_lbl_file.write_text("")
                n += 1

        print(f"  [{split}] {n}장 (COCO negative 없음)")

    # dataset yaml 생성
    yaml_path = config.DATA_DIR / "dataset_no_coco.yaml"
    yaml_path.write_text(f"""path: {NO_COCO_DIR}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: banner
""")
    print(f"\n[DONE] {yaml_path}")

if __name__ == "__main__":
    create_no_coco_dataset()
