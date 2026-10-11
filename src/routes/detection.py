from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from src.extensions import db
from src.models import DetectionRecord
from src.services.errors import DetectionError
from src.services.image_service import process_image
from src.services.video_service import process_video
from src.utils.file_utils import MediaFiles, ffmpeg_available
from src.utils.validators import validate_confidence

bp = Blueprint("detection", __name__)


def detection_page(kind):
    if request.method == "GET":
        return render_template("detect.html", kind=kind, ffmpeg_ready=ffmpeg_available(),
                               default_conf=current_app.config["YOLO_CONF"])
    files = MediaFiles(kind)
    try:
        detector = current_app.extensions.get("helmet_detector")
        if detector is None:
            raise DetectionError("模型尚未就绪，请配置 models/best.pt 并重新启动服务。")
        if kind == "video" and not ffmpeg_available():
            raise DetectionError("视频检测需要 ffmpeg，请先安装并重新尝试。")
        conf = validate_confidence(request.form.get("confidence"), current_app.config["YOLO_CONF"])
        name, original_path, source = files.save_upload(request.files.get("file"))
        current_app.logger.info("Detection started user=%s type=%s file=%s", current_user.id, kind, source.name)
        result_path, destination = files.result(".mp4" if kind == "video" else ".jpg")
        preview_path = None
        if kind == "video":
            preview_path, preview = files.result(".mp4", "preview")
            _, intermediate = files.result(".mp4", "intermediate")
            result = process_video(detector, source, destination, preview, intermediate, conf, current_app.config)
        else:
            result = process_image(detector, source, destination, conf)
        record = DetectionRecord(
            user_id=current_user.id, detection_type=kind, original_filename=name,
            original_path=original_path, result_path=result_path, preview_path=preview_path,
            total_objects=result["total"], class_counts=result["classes"], detections=result["detections"],
            processing_time=result["processing_time"], inference_time=result["inference_time"],
            confidence_threshold=conf, mean_confidence=result["mean_confidence"], media_info=result["media_info"],
        )
        db.session.add(record)
        db.session.commit()
        current_app.logger.info("Detection completed id=%s objects=%s seconds=%.3f", record.id, record.total_objects, record.processing_time)
        return redirect(url_for("history.detail", record_id=record.id))
    except DetectionError as error:
        db.session.rollback()
        files.cleanup()
        current_app.logger.warning("Detection failed user=%s type=%s: %s", current_user.id, kind, error, exc_info=True)
        flash(str(error), "error")
    except Exception:
        db.session.rollback()
        files.cleanup()
        current_app.logger.exception("Detection failed user=%s type=%s", current_user.id, kind)
        flash("检测失败，请检查文件后重新尝试；如果问题持续，请查看服务日志。", "error")
    return redirect(url_for("detection.video" if kind == "video" else "detection.image"))


@bp.route("/detect/image", methods=["GET", "POST"])
@bp.route("/image-detect", methods=["GET", "POST"])
@login_required
def image():
    return detection_page("image")


@bp.route("/detect/video", methods=["GET", "POST"])
@bp.route("/video-detect", methods=["GET", "POST"])
@login_required
def video():
    return detection_page("video")
