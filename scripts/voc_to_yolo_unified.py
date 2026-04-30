"""Build the unified RA3 YOLO dataset from PascalVOC XML files.

Input:
    data/datasets/ra3_unified/raw_frames/

Output:
    data/datasets/ra3_unified/{images,labels}/{train,val}/
    data/datasets/ra3_unified/dataset.yaml
"""
from __future__ import annotations

import csv
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "data" / "datasets" / "ra3_unified"
RAW_FRAMES = DATA_ROOT / "raw_frames"
CLASSES_PATH = DATA_ROOT / "predefined_classes.txt"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VAL_RATIO = 0.2
SEED = 42


def load_classes() -> list[str]:
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class file not found: {CLASSES_PATH}")
    classes = [line.strip() for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != len(set(classes)):
        duplicates = sorted({name for name in classes if classes.count(name) > 1})
        raise ValueError(f"Duplicate classes in {CLASSES_PATH}: {duplicates}")
    return classes


def load_sources_by_stem() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    out: dict[str, str] = {}
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            target_xml = row["target_xml"]
            if "\\" in target_xml:
                stem = PureWindowsPath(target_xml).stem
            else:
                stem = Path(target_xml).stem
            out[stem] = row["source"]
    return out


def find_image(stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = RAW_FRAMES / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_bbox(img_w: int, img_h: int, xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    xmin = max(0.0, min(float(img_w), xmin))
    ymin = max(0.0, min(float(img_h), ymin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymax = max(0.0, min(float(img_h), ymax))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bbox after clipping: {xmin}, {ymin}, {xmax}, {ymax}")
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def xml_to_yolo_lines(xml_path: Path, class_index: dict[str, int]) -> tuple[list[str], dict[str, int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    img_w = int(size.findtext("width") or "0")
    img_h = int(size.findtext("height") or "0")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {img_w}x{img_h}")

    lines: list[str] = []
    counts: dict[str, int] = {}
    for obj in root.findall("object"):
        cls_name = (obj.findtext("name") or "").strip()
        if cls_name not in class_index:
            raise ValueError(f"Unknown class {cls_name!r} in {xml_path}")
        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"Missing bndbox for {cls_name!r} in {xml_path}")
        cx, cy, w, h = convert_bbox(
            img_w,
            img_h,
            float(bbox.findtext("xmin") or "0"),
            float(bbox.findtext("ymin") or "0"),
            float(bbox.findtext("xmax") or "0"),
            float(bbox.findtext("ymax") or "0"),
        )
        cls_id = class_index[cls_name]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return lines, counts


def split_pairs(pairs: list[tuple[str, Path, Path]], sources_by_stem: dict[str, str]) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path, Path]]]:
    rng = random.Random(SEED)
    by_source: dict[str, list[tuple[str, Path, Path]]] = {}
    for pair in pairs:
        stem = pair[0]
        source = sources_by_stem.get(stem, "unknown")
        by_source.setdefault(source, []).append(pair)

    train: list[tuple[str, Path, Path]] = []
    val: list[tuple[str, Path, Path]] = []
    for source, group in sorted(by_source.items()):
        shuffled = group[:]
        rng.shuffle(shuffled)
        if len(shuffled) < 5:
            # Keep tiny sources, especially the single HUD screenshot, visible to training.
            train.extend(shuffled)
            continue
        n_val = max(1, round(len(shuffled) * VAL_RATIO))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def reset_output_dirs() -> None:
    for name in ("images", "labels"):
        target = DATA_ROOT / name
        if target.exists():
            resolved = target.resolve()
            root = DATA_ROOT.resolve()
            if root not in resolved.parents:
                raise RuntimeError(f"Refusing to delete outside dataset root: {target}")
            shutil.rmtree(target)
    for split in ("train", "val"):
        (DATA_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(classes: list[str]) -> None:
    lines = [
        f"path: {DATA_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(classes))
    (DATA_ROOT / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split(split: str, group: list[tuple[str, Path, Path]], class_index: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stem, xml_path, img_path in group:
        lines, local_counts = xml_to_yolo_lines(xml_path, class_index)
        img_dst = DATA_ROOT / "images" / split / f"{stem}{img_path.suffix.lower()}"
        lbl_dst = DATA_ROOT / "labels" / split / f"{stem}.txt"
        shutil.copy2(img_path, img_dst)
        lbl_dst.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        for name, count in local_counts.items():
            counts[name] = counts.get(name, 0) + count
    return counts


def main() -> None:
    classes = load_classes()
    class_index = {name: i for i, name in enumerate(classes)}
    sources_by_stem = load_sources_by_stem()

    xml_files = sorted(RAW_FRAMES.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {RAW_FRAMES}")

    pairs: list[tuple[str, Path, Path]] = []
    missing_images: list[Path] = []
    for xml_path in xml_files:
        img_path = find_image(xml_path.stem)
        if img_path is None:
            missing_images.append(xml_path)
            continue
        pairs.append((xml_path.stem, xml_path, img_path))

    if missing_images:
        raise FileNotFoundError(f"Images missing for {len(missing_images)} XML files, first: {missing_images[0]}")

    train_pairs, val_pairs = split_pairs(pairs, sources_by_stem)
    reset_output_dirs()
    train_counts = write_split("train", train_pairs, class_index)
    val_counts = write_split("val", val_pairs, class_index)
    write_dataset_yaml(classes)

    total_objects = sum(train_counts.values()) + sum(val_counts.values())
    print(f"Input pairs: {len(pairs)}")
    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}")
    print(f"Classes: {len(classes)}")
    print(f"Objects: {total_objects}")
    print(f"Output: {DATA_ROOT}")
    print("Source split:")
    for source in sorted(set(sources_by_stem.values()) | {"unknown"}):
        train_n = sum(1 for stem, _, _ in train_pairs if sources_by_stem.get(stem, "unknown") == source)
        val_n = sum(1 for stem, _, _ in val_pairs if sources_by_stem.get(stem, "unknown") == source)
        if train_n or val_n:
            print(f"  {source}: train={train_n}, val={val_n}")


if __name__ == "__main__":
    main()
