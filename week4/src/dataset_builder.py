"""Galaxy10 DECaLS 데이터셋을 YOLO 분류용 폴더 구조로 변환하는 모듈.

Hugging Face streaming 방식(1주차 Food101 심화 과제와 동일)으로 로드해
클래스당 per_class 장씩 샘플링하고, train/val 폴더로 나눠 저장한다.

YOLO 분류 모드가 요구하는 구조:
    data/galaxy10/
      train/<클래스명>/*.jpg
      val/<클래스명>/*.jpg
"""
from pathlib import Path

# Galaxy10 DECaLS 클래스 (라벨 0~9 순서)
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round_Smooth",
    "In-between_Round_Smooth",
    "Cigar_Shaped_Smooth",
    "Barred_Spiral",
    "Unbarred_Tight_Spiral",
    "Unbarred_Loose_Spiral",
    "Edge-on_without_Bulge",
    "Edge-on_with_Bulge",
]


def sanitize_class_name(name):
    """폴더명으로 안전한 클래스 이름을 만든다."""
    if not name:
        raise ValueError("클래스 이름이 비어 있습니다.")
    return name.strip().replace(" ", "_").replace("/", "-")


def assign_split(index_in_class, per_class, val_ratio=0.2):
    """클래스 내 몇 번째 샘플인지에 따라 train/val을 결정한다 (결정적 분할)."""
    if per_class <= 0:
        raise ValueError("per_class는 1 이상이어야 합니다.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")
    n_train = int(per_class * (1 - val_ratio))
    return "train" if index_in_class < n_train else "val"


def collection_done(counts, per_class):
    """모든 클래스가 per_class 장씩 모였는지 확인한다."""
    if counts is None:
        raise ValueError("counts가 None입니다.")
    return all(c >= per_class for c in counts.values())


def build_dataset(output_dir="data/galaxy10", per_class=200, val_ratio=0.2):
    """HF streaming으로 Galaxy10 DECaLS를 샘플링해 YOLO 분류용 폴더로 저장한다.

    Returns:
        저장된 이미지 총 수
    """
    from datasets import load_dataset  # 지연 임포트

    out = Path(output_dir)
    counts = {i: 0 for i in range(len(CLASS_NAMES))}

    ds = load_dataset("matthieulel/galaxy10_decals", split="train", streaming=True)
    saved = 0
    for sample in ds:
        label = int(sample["label"])
        if counts[label] >= per_class:
            if collection_done(counts, per_class):
                break
            continue

        split = assign_split(counts[label], per_class, val_ratio)
        cls_dir = out / split / sanitize_class_name(CLASS_NAMES[label])
        cls_dir.mkdir(parents=True, exist_ok=True)

        image = sample["image"]  # PIL Image
        image.convert("RGB").save(cls_dir / f"{CLASS_NAMES[label]}_{counts[label]:04d}.jpg")
        counts[label] += 1
        saved += 1
        if saved % 200 == 0:
            print(f"[dataset] {saved}장 저장... {dict(counts)}")

    print(f"[dataset] 완료 — 총 {saved}장 ({output_dir})")
    return saved
