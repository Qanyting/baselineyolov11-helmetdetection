import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest
import yaml

from scripts.prepare_shwd import convert_annotation, extract_archive, jpeg_needs_repair, prepare, sample_balanced, stratified_partition


def annotation(name="hat", coords=(1, 1, 10, 20), width=10, height=20):
    return f"""<annotation><size><width>{width}</width><height>{height}</height></size>
      <object><name>{name}</name><bndbox><xmin>{coords[0]}</xmin><ymin>{coords[1]}</ymin>
      <xmax>{coords[2]}</xmax><ymax>{coords[3]}</ymax></bndbox></object></annotation>"""


def test_voc_inclusive_pixel_conversion_and_head_mapping(tmp_path):
    xml = tmp_path / "box.xml"
    xml.write_text(annotation("person"))
    rows, counts, clipped = convert_annotation(xml, 10, 20)
    assert list(map(float, rows[0].split())) == [1, .5, .5, 1, 1]
    assert counts == {1: 1}
    assert clipped == 0


def test_missing_jpeg_end_marker_is_detected_without_changing_source(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (10, 20), "red").save(path)
    assert not jpeg_needs_repair(path)
    source = path.read_bytes()[:-2]
    path.write_bytes(source)
    assert jpeg_needs_repair(path)
    assert path.read_bytes() == source


@pytest.mark.parametrize("document", [annotation("worker"), annotation(coords=(8, 5, 3, 10)), annotation(width=30)])
def test_bad_annotations_fail_instead_of_silent_training(tmp_path, document):
    xml = tmp_path / "bad.xml"
    xml.write_text(document)
    with pytest.raises(ValueError):
        convert_annotation(xml, 10, 20)


def test_outside_box_is_clipped_to_image(tmp_path):
    xml = tmp_path / "clip.xml"
    xml.write_text(annotation(coords=(-2, -5, 100, 100)))
    rows, counts, clipped = convert_annotation(xml, 10, 20)
    assert list(map(float, rows[0].split())) == [0, .5, .5, 1, 1]
    assert clipped == 1


def test_split_reproducible_disjoint_and_sample_balances_images():
    records = [{"id": str(i), "counts": {0: 1} if i < 60 else {1: 20}} for i in range(100)]
    train, val = stratified_partition(records, 42)
    assert (train, val) == stratified_partition(list(reversed(records)), 42)
    assert len(train) == 80 and len(val) == 20
    assert {r["id"] for r in train}.isdisjoint(r["id"] for r in val)
    sampled = sample_balanced(train, 20, 42)
    assert sum(bool(r["counts"].get(0)) for r in sampled) == 10
    assert len({r["id"] for r in sampled}) == 20


def test_zip_traversal_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("good.txt", "safe")
        z.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "extracted/good.txt").exists()


def test_prepare_keeps_test_duplicate_out_of_fitting_and_resolves_yolo_paths(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("readme.txt", "fixture")
    voc = tmp_path / "extracted/VOC2028"
    for part in ["Annotations", "JPEGImages", "ImageSets/Main"]:
        (voc / part).mkdir(parents=True)
    ids = [f"image{i}" for i in range(12)]
    for i, image_id in enumerate(ids):
        color = 1 if i == 11 else i
        Image.new("RGB", (10, 20), (color * 10, 80, 20)).save(voc / "JPEGImages" / f"{image_id}.png")
        (voc / "Annotations" / f"{image_id}.xml").write_text(annotation("hat" if i % 2 else "person"))
    Image.new("RGB", (10, 20), (240, 240, 240)).save(voc / "JPEGImages/000377.png")
    (voc / "Annotations/000377.xml").write_text(annotation("dog"))
    (voc / "ImageSets/Main/trainval.txt").write_text("\n".join(ids[:10] + ["000377"]))
    (voc / "ImageSets/Main/test.txt").write_text("\n".join(ids[10:]))
    output = tmp_path / "yolo"
    result = prepare(archive, output, train_limit=4, val_limit=2)
    assert result["unique_images"] == 11
    assert result["excluded_images"][0]["id"] == "000377"
    assert not (output / "labels/000377.txt").exists()
    assert result["duplicates_removed"] == [{"removed": "image1", "kept": "image11"}]
    manifest = json.loads((output / "manifest.json").read_text())
    for splits in manifest["splits"].values():
        assert set(splits["train"]).isdisjoint(splits["val"])
        assert set(splits["test"]).isdisjoint(splits["train"] + splits["val"])
    config = yaml.safe_load((output / "m1.yaml").read_text())
    assert config["names"] == {0: "helmet", 1: "no_helmet"}
    for split in ("train", "val", "test"):
        for line in (output / config[split]).read_text().splitlines():
            image_path = Path(config["path"]) / line
            assert image_path.is_file()
            label_path = output / "labels" / f"{image_path.stem}.txt"
            assert label_path.is_file()
            values = list(map(float, label_path.read_text().split()))
            assert len(values) == 5 and all(0 < n <= 1 for n in values[1:])
