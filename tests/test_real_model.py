"""HELMET_TEST_IMAGE 指向本地有效图片时执行真实 YOLO + Web 全流程。"""
import os
from pathlib import Path

import cv2
import pytest

from src.extensions import db
from src.models import DetectionRecord
from src.services.detector import HelmetDetector
from tests.conftest import csrf_token, login, register


@pytest.fixture(scope="module")
def real_detector():
    if not Path("models/best.pt").is_file() or not os.getenv("HELMET_TEST_IMAGE"):
        pytest.skip("需要 models/best.pt 和 HELMET_TEST_IMAGE 才执行真实模型验收")
    return HelmetDetector("models/best.pt")


def test_real_yolo_web_image_and_video(app, client, real_detector, tmp_path):
    source = Path(os.environ["HELMET_TEST_IMAGE"])
    assert source.is_file()
    app.extensions["helmet_detector"] = real_detector
    model_identity = id(real_detector.model)
    register(client, "real_acceptance")
    login(client, "real_acceptance")
    token = csrf_token(client, "/detect/image")
    with source.open("rb") as file:
        response = client.post("/detect/image", data={"csrf_token": token, "file": (file, source.name)})
    assert response.location.startswith("/history/"), client.get(response.location).get_data(as_text=True)
    assert client.get(response.location).status_code == 200
    with app.app_context():
        image_record = db.session.scalar(db.select(DetectionRecord))
        assert sum(image_record.class_counts.values()) == image_record.total_objects
        assert set(image_record.class_counts) == set(real_detector.names.values())
        assert (Path(app.config["RESULT_FOLDER"]) / image_record.result_path).is_file()

    frame = cv2.imread(str(source))
    video = tmp_path / "real.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 3, (frame.shape[1], frame.shape[0]))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(frame)
    writer.release()
    token = csrf_token(client, "/detect/video")
    with video.open("rb") as file:
        response = client.post("/detect/video", data={"csrf_token": token, "file": (file, "real.avi")})
    assert response.location.startswith("/history/"), client.get(response.location).get_data(as_text=True)
    with app.app_context():
        record = db.session.scalar(db.select(DetectionRecord).where(DetectionRecord.detection_type == "video"))
        assert record.media_info["frame_count"] == 6
        assert sum(record.class_counts.values()) == record.total_objects
        result_video = Path(app.config["RESULT_FOLDER"]) / record.result_path
        capture = cv2.VideoCapture(str(result_video))
        assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
        capture.release()
    assert id(real_detector.model) == model_identity


def test_missing_weight_does_not_download_or_crash_app(tmp_path):
    from src import create_app
    app = create_app({"TESTING": True, "SECRET_KEY": "missing-model-test",
                      "MODEL_PATH": tmp_path / "missing.pt", "INITIALIZE_DETECTOR": True,
                      "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                      "UPLOAD_FOLDER": tmp_path / "upload", "RESULT_FOLDER": tmp_path / "result"})
    assert app.test_client().get("/login").status_code == 200
    assert "YOLO model not found" in app.extensions["model_error"]
    assert not (tmp_path / "missing.pt").exists()
