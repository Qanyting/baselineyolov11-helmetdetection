"""保存一次可复现的真实模型诊断结果，与业务上传和历史记录分开。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.detector import HelmetDetector
from src.services.errors import DetectionError


def main():
    parser = argparse.ArgumentParser(description="实际调用模型，输出标注图片和 JSON；不会修改平台的权重或配置。")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "test-artifacts/diagnostic")
    args = parser.parse_args()
    if not 0.05 <= args.conf <= 0.95:
        parser.error("--conf 需在 0.05–0.95 范围内")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        detector = HelmetDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
        result = detector.detect_image(args.image, args.output / "annotated.jpg")
    except (DetectionError, OSError) as error:
        parser.exit(1, f"检测失败：{error}\n")
    report = {"image": str(args.image.resolve()), "model": str(args.model.resolve()),
              "device": detector.device, "conf": args.conf, "imgsz": args.imgsz,
              "model_metadata": detector.metadata, "result": result}
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"classes": result["classes"], "total": result["total"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
