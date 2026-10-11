import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.services.errors import DetectionError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


def validate_upload_name(filename, kind):
    if not filename or not filename.strip():
        raise DetectionError("请先选择一个文件。")
    extension = Path(filename).suffix.lower()
    allowed = VIDEO_EXTENSIONS if kind == "video" else IMAGE_EXTENSIONS
    if extension not in allowed:
        raise DetectionError("文件格式不支持，请选择页面列出的图片或视频格式。")
    return extension


def read_image(path, max_pixels=25_000_000):
    """验证真实格式、尺寸及完整性；推理前按 EXIF 校正方向。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != FORMATS.get(Path(path).suffix.lower()):
                    raise DetectionError("图片内容与扩展名不符，请重新导出图片后上传。")
                if image.width * image.height > max_pixels:
                    raise DetectionError(f"图片像素过大，请缩小至 {max_pixels // 1_000_000} 百万像素以内。")
                image.verify()
            with Image.open(path) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except DetectionError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DetectionError("图片已损坏或无法解码，请换一张有效图片。") from error


def validate_confidence(value, default):
    try:
        conf = float(value) if value else default
    except (TypeError, ValueError) as error:
        raise DetectionError("置信度阈值需为 0.05–0.95 之间的数字。") from error
    if not 0.05 <= conf <= 0.95:
        raise DetectionError("置信度阈值需在 0.05–0.95 之间。")
    return conf
