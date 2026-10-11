"""Evaluate a trained checkpoint on the preserved SHWD test split; optionally inspect the user's image."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/helmet_shwd_m1/weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/shwd/m1.yaml")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if not args.model.is_file() or not args.data.is_file():
        parser.error("Model or dataset is missing. Finish training first.")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from ultralytics import YOLO
    model = YOLO(str(args.model.resolve()))
    if model.task != "detect" or model.names != {0: "helmet", 1: "no_helmet"}:
        parser.error("Expected a helmet/no_helmet detection checkpoint")
    metrics = model.val(data=str(args.data.resolve()), split="test", device=args.device,
                        imgsz=args.imgsz, batch=1, workers=0, plots=True,
                        project=str(ROOT / "runs"), name="helmet_test", exist_ok=False)
    with args.model.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    report = {"model": str(args.model.resolve()), "model_sha256": digest,
              "data": str(args.data.resolve()), "split": "test", "imgsz": args.imgsz,
              "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
              "classes": {model.names[int(i)]: dict(zip(["precision", "recall", "mAP50", "mAP50_95"],
                            map(float, metrics.box.class_result(index))))
                          for index, i in enumerate(metrics.box.ap_class_index)}}
    output = Path(metrics.save_dir)
    if args.image:
        model.predict(source=str(args.image.resolve()), device=args.device, imgsz=640, conf=.25,
                      save=True, save_txt=True, save_conf=True, project=str(output), name="sample", exist_ok=True)
        report["sample_image"] = str(args.image.resolve())
        report["sample_note"] = "Visual comparison only; the image has no ground-truth annotations."
    path = output / "evaluation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Evaluation report: {path}")
    print("Inspect test metrics and the image result before installing the checkpoint.")


if __name__ == "__main__":
    main()
