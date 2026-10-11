"""集中配置；相对路径始终以项目根目录为基准。"""
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def database_uri():
    value = os.getenv("DATABASE_URI", "sqlite:///database/app.db")
    if value.startswith("sqlite:///") and value != "sqlite:///:memory:":
        return "sqlite:///" + str(project_path(value.removeprefix("sqlite:///")))
    return value


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or None
    SQLALCHEMY_DATABASE_URI = database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_PATH = project_path(os.getenv("MODEL_PATH", "models/best.pt"))
    INITIALIZE_DETECTOR = True
    UPLOAD_FOLDER = project_path(os.getenv("UPLOAD_FOLDER", "storage/uploads"))
    RESULT_FOLDER = project_path(os.getenv("RESULT_FOLDER", "storage/results"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
    MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "600"))
    MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "18000"))
    YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
    YOLO_IOU = float(os.getenv("YOLO_IOU", "0.7"))
    YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")
    YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
    FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "ffmpeg")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    REMEMBER_COOKIE_SECURE = SESSION_COOKIE_SECURE
    REMEMBER_COOKIE_DURATION = timedelta(days=30)
    WTF_CSRF_TIME_LIMIT = 12 * 60 * 60
