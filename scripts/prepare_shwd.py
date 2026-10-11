"""Download official SHWD, validate VOC annotations, and create reproducible YOLO splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset"
DOWNLOAD = "https://drive.usercontent.google.com/download?id=1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX&export=download&confirm=t"
ARCHIVE_SHA256 = "2f6c5c347b778271bc8b27c2d61b731fc5fb8cdb4f45fce583dd2f82e6a46401"
CLASSES = {"hat": 0, "person": 1}  # In SHWD 'person' means an unhelmeted HEAD.
EXCLUDED_IMAGES = {"000377": "Source XML has three incorrect dog labels on helmet regions; exclude the whole collage rather than guess corrected labels."}


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jpeg_needs_repair(path):
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return False
    with path.open("rb") as f:
        f.seek(-2, 2)
        return f.read() != b"\xff\xd9"


def download_archive(path):
    if path.is_file() and zipfile.is_zipfile(path):
        print(f"Using existing archive: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".zip.part")
    with requests.get(DOWNLOAD, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        received, started = 0, time.monotonic()
        with temporary.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                received += len(chunk)
                if time.monotonic() - started > 15:
                    print(f"Download: {received / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                    started = time.monotonic()
        if total and received != total:
            raise ValueError("Download is incomplete; rerun preparation.")
    if not zipfile.is_zipfile(temporary):
        raise ValueError("Download returned a page instead of the dataset ZIP.")
    temporary.replace(path)


def extract_archive(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            candidate = (target / item.filename).resolve()
            if not candidate.is_relative_to(target.resolve()):
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink is unsupported: {item.filename}")
        z.extractall(target)


def convert_annotation(xml, width, height):
    tree = ET.parse(xml).getroot()
    declared = tree.find("size")
    if declared is None or (int(declared.findtext("width")), int(declared.findtext("height"))) != (width, height):
        raise ValueError(f"Image and annotation dimensions differ: {xml}")
    lines, counts, clipped = [], Counter(), 0
    for obj in tree.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            raise ValueError(f"Unknown class {name!r}: {xml}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing box: {xml}")
        # Pascal VOC uses 1-based inclusive pixels; convert to 0-based edges.
        raw = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not all(math.isfinite(value) for value in raw):
            raise ValueError(f"Non-finite bounding box: {xml}")
        edges = [raw[0] - 1, raw[1] - 1, raw[2], raw[3]]
        x1, y1, x2, y2 = max(0, edges[0]), max(0, edges[1]), min(width, edges[2]), min(height, edges[3])
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate bounding box: {xml}: {raw}")
        clipped += [x1, y1, x2, y2] != edges
        label = CLASSES[name]
        lines.append(f"{label} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
        counts[label] += 1
    if not lines:
        raise ValueError(f"Empty annotation: {xml}")
    return lines, counts, clipped


def stratified_partition(records, seed):
    groups = defaultdict(list)
    for item in records:
        groups[bool(item["counts"].get(0))].append(item)
    train, val = [], []
    rng = random.Random(seed)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item["id"])
        rng.shuffle(group)
        count = max(1, round(len(group) * .2))
        val.extend(group[:count])
        train.extend(group[count:])
    return train, val


def sample_balanced(records, limit, seed):
    if len(records) <= limit:
        return records
    groups = defaultdict(list)
    for item in records:
        groups[bool(item["counts"].get(0))].append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Prefer equal image counts with/without helmets rather than the source's dense-head imbalance.
    selected = []
    for index in range(max(map(len, groups.values()))):
        for key in sorted(groups, reverse=True):
            if index < len(groups[key]):
                selected.append(groups[key][index])
                if len(selected) == limit:
                    return selected
    return selected


def prepare(archive, output, seed=42, train_limit=2000, val_limit=400):
    raw = archive.parent / "extracted"
    if not list(raw.glob("**/Annotations/*.xml")):
        print("Extracting dataset…", flush=True)
        extract_archive(archive, raw)
    candidates = list(raw.glob("**/Annotations"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one VOC annotation directory, found {candidates}")
    voc = candidates[0].parent
    image_sets = voc / "ImageSets/Main"
    train_ids = set((image_sets / "trainval.txt").read_text().split())
    test_ids = set((image_sets / "test.txt").read_text().split())
    if train_ids & test_ids:
        raise ValueError("Official trainval and test IDs overlap")
    images = {p.stem: p for p in (voc / "JPEGImages").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    annotations = {p.stem: p for p in (voc / "Annotations").glob("*.xml")}
    if set(images) != set(annotations) or set(images) != train_ids | test_ids:
        raise ValueError("Image, annotation, and official split inventories differ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    records, seen, duplicates, clipped_total, excluded, repairs = [], {}, [], 0, [], []
    # Keep official test first so an exact duplicate can never leak into fitting.
    order = sorted(test_ids) + sorted(train_ids)
    for index, image_id in enumerate(order):
        if image_id in EXCLUDED_IMAGES:
            excluded.append({"id": image_id, "reason": EXCLUDED_IMAGES[image_id]})
            continue
        source = images[image_id]
        with Image.open(source) as image:
            image.load()
            if image.getexif().get(274, 1) not in (0, 1):
                raise ValueError(f"Rotated source needs explicit annotation correction: {source}")
            width, height = image.size
            decoded = image.convert("RGB")
            digest = hashlib.sha256(f"{width}x{height}:".encode() + decoded.tobytes()).hexdigest()
        lines, counts, clipped = convert_annotation(annotations[image_id], width, height)
        clipped_total += clipped
        if digest in seen:
            duplicates.append({"removed": image_id, "kept": seen[digest]})
            continue
        seen[digest] = image_id
        target = output / "images" / source.name
        if not target.exists():
            shutil.copy2(source, target)
        if jpeg_needs_repair(source):
            # Match Ultralytics' JPEG repair while keeping the official archive/extracted source intact.
            if jpeg_needs_repair(target):
                decoded.save(target, "JPEG", subsampling=0, quality=100)
            repairs.append(image_id)
        (output / "labels" / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
        records.append({"id": image_id, "image": source.name, "hash": digest, "counts": dict(counts),
                        "official_split": "test" if image_id in test_ids else "trainval"})
        if (index + 1) % 500 == 0:
            print(f"Validated {index + 1}/{len(order)} images", flush=True)
    test = [r for r in records if r["official_split"] == "test"]
    train, val = stratified_partition([r for r in records if r["official_split"] == "trainval"], seed)
    profiles = {"full": {"train": train, "val": val, "test": test},
                "m1": {"train": sample_balanced(train, train_limit, seed),
                       "val": sample_balanced(val, val_limit, seed + 1), "test": test}}
    summary = {"source": SOURCE, "download": DOWNLOAD, "archive_sha256": sha256(archive),
               "seed": seed, "source_images": len(order), "unique_images": len(records),
               "excluded_images": excluded,
               "jpeg_end_marker_repairs": repairs,
               "duplicates_removed": duplicates, "clipped_boxes": clipped_total,
               "mapping": {"hat": "0: helmet", "person (bare head)": "1: no_helmet"}, "profiles": {}}
    for name, splits in profiles.items():
        profile_summary = {}
        for split, subset in splits.items():
            listing = output / f"{name}_{split}.txt"
            listing.write_text("".join(f"./images/{r['image']}\n" for r in subset))
            counts = Counter()
            for item in subset:
                counts.update(item["counts"])
            profile_summary[split] = {"images": len(subset), "helmet_boxes": counts[0], "no_helmet_boxes": counts[1]}
        data = {"path": str(output.resolve()), **{s: f"{name}_{s}.txt" for s in splits},
                "names": {0: "helmet", 1: "no_helmet"}}
        (output / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        summary["profiles"][name] = profile_summary
    (output / "manifest.json").write_text(json.dumps({"summary": summary, "records": records,
        "splits": {p: {s: [r["id"] for r in subset] for s, subset in splits.items()} for p, splits in profiles.items()}}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "datasets/raw/VOC2028.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/shwd")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_limit, args.val_limit) < 2:
        parser.error("Dataset limits must be at least two images")
    download_archive(args.archive)
    if sha256(args.archive) != ARCHIVE_SHA256:
        parser.error("Archive differs from the checked 2026-10-06 SHWD download; verify the source before converting.")
    prepare(args.archive, args.output, args.seed, args.train_limit, args.val_limit)


if __name__ == "__main__":
    main()
