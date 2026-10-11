import re
import shutil
from pathlib import Path
from uuid import uuid4

from flask import current_app

from src.services.errors import DetectionError
from src.utils.validators import validate_upload_name


def display_filename(name):
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"[\x00-\x1f\x7f]", "", basename)[:255] or "上传文件"


class MediaFiles:
    """跟踪一次检测产生的文件，失败时统一清理。路径均相对受保护的存储根。"""

    def __init__(self, kind):
        self.kind = kind
        self.subfolder = "videos" if kind == "video" else "images"
        self.identifier = uuid4().hex
        self.created = []

    def result(self, suffix, label="result"):
        relative = f"{self.subfolder}/{self.identifier}_{label}{suffix}"
        absolute = storage_path("RESULT_FOLDER", relative)
        self.created.append(absolute)
        return relative, absolute

    def save_upload(self, upload):
        if upload is None:
            raise DetectionError("请先选择一个文件。")
        extension = validate_upload_name(upload.filename, self.kind)
        relative = f"{self.subfolder}/{self.identifier}{extension}"
        absolute = storage_path("UPLOAD_FOLDER", relative)
        self.created.append(absolute)
        upload.save(absolute)
        if absolute.stat().st_size == 0:
            raise DetectionError("上传的是空文件，请重新选择。")
        if absolute.stat().st_size > current_app.config["MAX_CONTENT_LENGTH"]:
            raise DetectionError("上传文件超过大小限制。")
        return display_filename(upload.filename), relative, absolute

    def cleanup(self):
        for path in self.created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                current_app.logger.exception("Failed to clean up file %s", path)


def storage_path(key, relative):
    root = Path(current_app.config[key]).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise DetectionError("文件路径无效。")
    return target


def ffmpeg_available():
    return shutil.which(current_app.config["FFMPEG_BINARY"]) is not None
