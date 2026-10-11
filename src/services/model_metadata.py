"""模型说明与权重指纹绑定，替换权重后不会沿用旧模型的说明。"""
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_model_metadata(model_path):
    path = Path(model_path)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if not isinstance(metadata, dict) or metadata.get("sha256") != digest:
            logger.warning("Model metadata fingerprint does not match %s", path)
            return {}
        return {key: value for key, value in metadata.items()
                if key in {"title", "training_domain", "notice"} and isinstance(value, str)}
    except (OSError, ValueError):
        logger.warning("Cannot read model metadata %s", sidecar, exc_info=True)
        return {}
