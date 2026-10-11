"""Install an explicitly selected, evaluated checkpoint with a backup of the current platform model."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/helmet_shwd_m1/weights/best.pt")
    parser.add_argument("--report", type=Path, required=True, help="evaluation.json from evaluate_helmet.py")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    fingerprint = digest(args.model)
    if report.get("model_sha256") != fingerprint or report.get("split") != "test":
        parser.error("The evaluation report does not match this model and its test evaluation.")
    from ultralytics import YOLO
    model = YOLO(str(args.model.resolve()))
    if model.task != "detect" or model.names != {0: "helmet", 1: "no_helmet"}:
        parser.error("Expected 0=helmet, 1=no_helmet detection checkpoint")
    target = ROOT / "models/best.pt"
    backup = ROOT / "models/backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True)
    for previous in (target, target.with_suffix(".json")):
        if previous.exists():
            shutil.copy2(previous, backup / previous.name)
    metadata = {"sha256": fingerprint, "title": "YOLO11n 安全帽佩戴模型",
                "training_domain": "SHWD 安全帽与未戴帽头部",
                "notice": "该模型已在 SHWD 测试集评估；新场景和遮挡仍可能产生误检或漏检，请核对结果。",
                "evaluation": report}
    temporary = target.with_suffix(".pt.new")
    shutil.copy2(args.model, temporary)
    temporary.replace(target)
    target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Installed: {target}\nPrevious model backup: {backup}\nRestart python app.py to load it.")


if __name__ == "__main__":
    main()
