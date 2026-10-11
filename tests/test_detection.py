import io
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.extensions import db
from src.models import DetectionRecord
from src.services.errors import DetectionError
from src.utils.validators import read_image
from tests.conftest import csrf_token, login, register


class StubDetector:
    """隔离 Web 测试中的硬件依赖；真实模型另有集成测试。"""
    names = {0: "custom_helmet", 1: "custom_uncovered"}
    device = "test"

    def detect_frame(self, frame, conf=None):
        detection = {"class_id": 0, "class_name": "custom_helmet", "confidence": 0.9,
                     "bbox": [2, 2, 28, 28]}
        return {"total": 1, "classes": {"custom_helmet": 1, "custom_uncovered": 0},
                "detections": [detection], "inference_time": 0.01, "mean_confidence": 0.9}

    def annotate(self, frame, detections):
        output = frame.copy()
        cv2.rectangle(output, (2, 2), (28, 28), (112, 137, 47), 2)
        return output

    def detect_image(self, source, destination, conf):
        frame = read_image(source)
        result = self.detect_frame(frame, conf)
        assert cv2.imwrite(str(destination), self.annotate(frame, result["detections"]))
        result["media_info"] = {"width": frame.shape[1], "height": frame.shape[0]}
        return result


@pytest.fixture
def signed_in(app, client):
    app.extensions["helmet_detector"] = StubDetector()
    register(client)
    login(client)
    return client


def image_bytes(extension=".png"):
    success, encoded = cv2.imencode(extension, np.full((64, 80, 3), 170, dtype=np.uint8))
    assert success
    return encoded.tobytes()


def upload_image(client, filename="现场.png", contents=None, confidence="0.25"):
    return client.post("/detect/image", data={"csrf_token": csrf_token(client, "/detect/image"),
                       "confidence": confidence,
                       "file": (io.BytesIO(image_bytes() if contents is None else contents), filename)})


def record(app):
    with app.app_context():
        return db.session.scalar(db.select(DetectionRecord))


def stored_files(app):
    return [path for key in ("UPLOAD_FOLDER", "RESULT_FOLDER")
            for path in Path(app.config[key]).rglob("*") if path.is_file()]


def test_image_record_result_dynamic_classes_and_private_media(app, signed_in):
    detector = app.extensions["helmet_detector"]
    response = upload_image(signed_in, "../../现场.png")
    assert response.status_code == 302
    item = record(app)
    assert response.location == f"/history/{item.id}"
    assert item.original_filename == "现场.png"
    assert ".." not in item.original_path
    assert item.class_counts == {"custom_helmet": 1, "custom_uncovered": 0}
    assert len(stored_files(app)) == 2
    detail = signed_in.get(response.location).get_data(as_text=True)
    assert "custom_helmet" in detail and "90.0%" in detail
    assert signed_in.get("/history").status_code == 200
    original = signed_in.get(f"/media/{item.id}/original")
    assert original.status_code == 200 and original.data == image_bytes()
    assert original.headers["Cache-Control"] == "private, no-store"
    assert signed_in.get(f"/media/{item.id}/result?download=1").headers["Content-Disposition"].startswith("attachment;")
    assert app.extensions["helmet_detector"] is detector
    assert signed_in.get(f"/static/uploads/{item.original_path}").status_code == 404


@pytest.mark.parametrize("filename,contents,confidence", [
    ("bad.exe", b"malware", "0.25"), ("bad.png", b"not an image", "0.25"),
    ("empty.png", b"", "0.25"), ("wrong.jpg", image_bytes(), "0.25"),
    ("valid.png", image_bytes(), "nan"), ("valid.png", image_bytes(), "0.99"),
])
def test_invalid_uploads_leave_no_records_or_files(app, signed_in, filename, contents, confidence):
    response = upload_image(signed_in, filename, contents, confidence)
    assert response.status_code == 302
    assert record(app) is None
    assert not stored_files(app)
    assert "Traceback" not in signed_in.get(response.location).get_data(as_text=True)


def test_model_missing_is_actionable_and_does_not_save_upload(app, signed_in):
    app.extensions.pop("helmet_detector")
    response = upload_image(signed_in)
    page = signed_in.get(response.location).get_data(as_text=True)
    assert "模型尚未就绪" in page
    assert not stored_files(app)


def test_size_limit_returns_413_without_persisting_files(app, signed_in):
    token = csrf_token(signed_in, "/detect/image")
    app.config["MAX_CONTENT_LENGTH"] = 1024
    response = signed_in.post("/detect/image", data={"csrf_token": token,
                              "file": (io.BytesIO(b"x" * 2048), "large.png")})
    assert response.status_code == 413
    assert not stored_files(app)


def test_inference_failure_and_database_failure_cleanup(app, signed_in, monkeypatch):
    def fail(*args, **kwargs):
        raise DetectionError("模型推理失败")
    with monkeypatch.context() as patch:
        patch.setattr(app.extensions["helmet_detector"], "detect_image", fail)
        assert upload_image(signed_in).status_code == 302
        assert record(app) is None and not stored_files(app)
    token = csrf_token(signed_in, "/detect/image")
    def fail_commit():
        raise SQLAlchemyError("forced database failure")
    with monkeypatch.context() as patch:
        patch.setattr(db.session, "commit", fail_commit)
        response = signed_in.post("/detect/image", data={"csrf_token": token,
                                  "file": (io.BytesIO(image_bytes()), "valid.png")})
        assert response.status_code == 302
    assert record(app) is None and not stored_files(app)


def test_user_isolation_for_details_media_delete_and_history(app, signed_in):
    upload_image(signed_in, "only-first-user.png")
    item = record(app)
    other = app.test_client()
    register(other, "second_user")
    login(other, "second_user")
    for path in (f"/history/{item.id}", f"/media/{item.id}/original", f"/media/{item.id}/result"):
        assert other.get(path).status_code == 404
    assert "only-first-user.png" not in other.get("/history").get_data(as_text=True)
    token = csrf_token(other, "/history")
    assert other.post(f"/history/{item.id}/delete", data={"csrf_token": token}).status_code == 404
    assert record(app) is not None and len(stored_files(app)) == 2


def test_delete_requires_post_and_csrf_and_removes_all_files(app, signed_in):
    upload_image(signed_in)
    item = record(app)
    assert signed_in.get(f"/history/{item.id}/delete").status_code == 405
    assert signed_in.post(f"/history/{item.id}/delete").status_code == 400
    token = csrf_token(signed_in, f"/history/{item.id}")
    response = signed_in.post(f"/history/{item.id}/delete", data={"csrf_token": token})
    assert response.status_code == 302
    assert record(app) is None and not stored_files(app)


def test_delete_database_failure_restores_files(app, signed_in, monkeypatch):
    upload_image(signed_in)
    item = record(app)
    paths = stored_files(app)
    token = csrf_token(signed_in, f"/history/{item.id}")
    def fail_commit():
        raise SQLAlchemyError("forced failure")
    with monkeypatch.context() as patch:
        patch.setattr(db.session, "commit", fail_commit)
        response = signed_in.post(f"/history/{item.id}/delete", data={"csrf_token": token})
    assert response.location == f"/history/{item.id}"
    assert record(app) is not None
    assert all(path.exists() for path in paths)


def test_delete_file_cleanup_failure_is_reported(app, signed_in, monkeypatch):
    upload_image(signed_in)
    item = record(app)
    token = csrf_token(signed_in, f"/history/{item.id}")
    real_unlink = Path.unlink
    def blocked_unlink(path, *args, **kwargs):
        if path.name.startswith(".delete-"):
            raise PermissionError("simulated storage permission failure")
        return real_unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", blocked_unlink)
        response = signed_in.post(f"/history/{item.id}/delete", data={"csrf_token": token})
    assert record(app) is None
    page = signed_in.get(response.location).get_data(as_text=True)
    assert "部分本地文件未能清理" in page
    assert "检测记录及对应文件已删除" not in page


def test_history_filtering_and_pagination(app, signed_in):
    for index in range(13):
        upload_image(signed_in, f"image-{index}.png")
    first = signed_in.get("/history").get_data(as_text=True)
    assert "第 1 / 2 页" in first and "image-12.png" in first
    second = signed_in.get("/history?page=2").get_data(as_text=True)
    assert "image-0.png" in second
    video = signed_in.get("/history?type=video").get_data(as_text=True)
    assert "共 0 条记录" in video


def make_video(path, frames=6):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 3, (80, 64))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((64, 80, 3), 170, dtype=np.uint8))
    writer.release()


def test_video_pipeline_h264_preview_range_requests_and_delete(app, signed_in, tmp_path):
    source = tmp_path / "sample.avi"
    make_video(source)
    token = csrf_token(signed_in, "/detect/video")
    with source.open("rb") as file:
        response = signed_in.post("/detect/video", data={"csrf_token": token, "file": (file, "sample.avi")})
    item = record(app)
    assert item is not None, signed_in.get(response.location).get_data(as_text=True)
    assert item.is_video and item.total_objects == 6
    assert item.media_info["frame_count"] == 6 and item.media_info["duration"] == 2
    assert len(stored_files(app)) == 3
    result = Path(app.config["RESULT_FOLDER"]) / item.result_path
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate", "-of", "json", str(result)]))
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert stream["width"] == 80 and stream["height"] == 64 and stream["r_frame_rate"] == "3/1"
    assert signed_in.get(f"/media/{item.id}/result", headers={"Range": "bytes=0-255"}).status_code == 206
    assert signed_in.get(f"/media/{item.id}/preview").status_code == 200
    page = signed_in.get(response.location).get_data(as_text=True)
    assert "不代表唯一人员数量" in page
    token = csrf_token(signed_in, response.location)
    signed_in.post(f"/history/{item.id}/delete", data={"csrf_token": token})
    assert not stored_files(app)


def test_broken_video_and_missing_ffmpeg_cleanup(app, signed_in):
    token = csrf_token(signed_in, "/detect/video")
    signed_in.post("/detect/video", data={"csrf_token": token, "file": (io.BytesIO(b"bad video"), "bad.mp4")})
    assert record(app) is None and not stored_files(app)
    app.config["FFMPEG_BINARY"] = "/does/not/exist"
    token = csrf_token(signed_in, "/detect/video")
    response = signed_in.post("/detect/video", data={"csrf_token": token, "file": (io.BytesIO(b"bad video"), "bad.mp4")})
    assert "视频检测需要 ffmpeg" in signed_in.get(response.location).get_data(as_text=True)
    assert not stored_files(app)


def test_video_limits_and_transcode_failure_cleanup(app, signed_in, tmp_path, monkeypatch):
    source = tmp_path / "limit.avi"
    make_video(source)
    app.config["MAX_VIDEO_FRAMES"] = 3
    token = csrf_token(signed_in, "/detect/video")
    with source.open("rb") as file:
        response = signed_in.post("/detect/video", data={"csrf_token": token, "file": (file, "limit.avi")})
    assert "视频超过处理时长或帧数上限" in signed_in.get(response.location).get_data(as_text=True)
    assert record(app) is None and not stored_files(app)
    app.config["MAX_VIDEO_FRAMES"] = 100
    def failed_transcode(*args, **kwargs):
        raise DetectionError("视频转码失败")
    monkeypatch.setattr("src.services.video_service.transcode", failed_transcode)
    token = csrf_token(signed_in, "/detect/video")
    with source.open("rb") as file:
        response = signed_in.post("/detect/video", data={"csrf_token": token, "file": (file, "limit.avi")})
    assert "视频转码失败" in signed_in.get(response.location).get_data(as_text=True)
    assert record(app) is None and not stored_files(app)


def test_no_objects_is_saved_as_success(app, signed_in, monkeypatch):
    def no_objects(frame, conf=None):
        return {"total": 0, "classes": {"custom_helmet": 0, "custom_uncovered": 0},
                "detections": [], "inference_time": 0.01, "mean_confidence": None}
    monkeypatch.setattr(app.extensions["helmet_detector"], "detect_frame", no_objects)
    response = upload_image(signed_in)
    item = record(app)
    assert item.total_objects == 0 and item.mean_confidence is None
    assert "当前阈值下未识别到目标" in signed_in.get(response.location).get_data(as_text=True)


def test_image_pixel_limit_and_storage_path_guard(tmp_path, app):
    from src.utils.file_utils import storage_path
    image = tmp_path / "large.png"
    image.write_bytes(image_bytes())
    with pytest.raises(DetectionError, match="图片像素过大"):
        read_image(image, max_pixels=100)
    with app.app_context(), pytest.raises(DetectionError, match="文件路径无效"):
        storage_path("UPLOAD_FOLDER", "../../secret")
