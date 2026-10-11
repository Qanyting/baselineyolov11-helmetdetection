"""Train YOLO11n on SHWD with conservative defaults for an M1 / 8GB Mac."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_settings(args):
    return dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, cache=False, amp=False,
                project=str(ROOT / "runs"), name=args.name, exist_ok=False,
                optimizer="AdamW", lr0=0.001, patience=12, seed=42,
                deterministic=True, mosaic=0.5, mixup=0.0, close_mosaic=5,
                plots=True, save=True, save_period=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/shwd/m1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models/pretrained/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--name", default="helmet_shwd_m1")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from weights/last.pt")
    parser.add_argument("--check", action="store_true", help="Check data, model and settings; do not train")
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.imgsz) <= 0 or args.imgsz % 32:
        parser.error("epochs/batch must be positive; imgsz must be a positive multiple of 32")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a directory name")
    # Enable PyTorch's CPU fallback only for MPS operations without a native implementation.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu")
    source = args.resume or args.model
    if not source.is_file() or not args.data.is_file():
        parser.error("Missing model or dataset. Run scripts/prepare_shwd.py first.")
    data = check_det_dataset(str(args.data.resolve()), autodownload=False)
    if data["names"] != {0: "helmet", 1: "no_helmet"}:
        parser.error("Dataset must use 0=helmet, 1=no_helmet")
    model = YOLO(str(source.resolve()))
    if model.task != "detect":
        parser.error("A detection checkpoint is required")
    settings = build_settings(args)
    get_cfg(overrides=settings)  # Validate options against the actually installed Ultralytics.
    if args.resume:
        if model.names != data["names"]:
            parser.error("Resume checkpoint has incompatible classes")
        settings = dict(resume=True, device=args.device, batch=args.batch, workers=0,
                        cache=False, amp=False, imgsz=args.imgsz)
        print("Resume restores original data, epochs and run directory from last.pt.")
    elif (ROOT / "runs" / args.name).exists() and not args.check:
        parser.error("Run directory already exists. Use --resume .../weights/last.pt or a new --name.")
    print(json.dumps({"model": str(source.resolve()), "settings": settings,
                      "mode": "check_only" if args.check else "train"}, indent=2))
    if args.check:
        from ultralytics.data.dataset import YOLODataset
        for split in ("train", "val", "test"):
            dataset = YOLODataset(img_path=data[split], data=data, imgsz=args.imgsz,
                                  augment=False, cache=False, batch_size=args.batch)
            if not dataset.labels:
                parser.error(f"Empty {split} dataset")
            print(f"{split}: {len(dataset)} images, {sum(len(label['cls']) for label in dataset.labels)} boxes")
        print("Preflight passed. No training has been started.")
        return
    print("Starting training. The platform's models/best.pt will not be replaced.", flush=True)
    model.train(**settings)


if __name__ == "__main__":
    main()
