from uuid import uuid4

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required

from src.extensions import db
from src.models import DetectionRecord
from src.utils.file_utils import storage_path

bp = Blueprint("history", __name__)


def own_record(record_id):
    record = db.session.scalar(db.select(DetectionRecord).where(
        DetectionRecord.id == record_id, DetectionRecord.user_id == current_user.id
    ))
    if record is None:
        abort(404)
    return record


@bp.get("/history")
@login_required
def index():
    query = db.select(DetectionRecord).where(DetectionRecord.user_id == current_user.id)
    kind = request.args.get("type", "all")
    if kind in {"image", "video"}:
        query = query.where(DetectionRecord.detection_type == kind)
    else:
        kind = "all"
    pagination = db.paginate(query.order_by(DetectionRecord.created_at.desc(), DetectionRecord.id.desc()), per_page=12, error_out=False)
    return render_template("history.html", pagination=pagination, kind=kind)


@bp.get("/history/<int:record_id>")
@login_required
def detail(record_id):
    return render_template("result.html", record=own_record(record_id))


@bp.get("/media/<int:record_id>/<variant>")
@login_required
def media(record_id, variant):
    record = own_record(record_id)
    if variant == "original":
        key, relative, name = "UPLOAD_FOLDER", record.original_path, record.original_filename
    elif variant == "result":
        key, relative, name = "RESULT_FOLDER", record.result_path, f"detected_{record.id}.{'mp4' if record.is_video else 'jpg'}"
    elif variant == "preview" and record.preview_path:
        key, relative, name = "RESULT_FOLDER", record.preview_path, f"preview_{record.id}.mp4"
    else:
        abort(404)
    path = storage_path(key, relative)
    if not path.is_file():
        abort(404)
    return send_file(path, conditional=True, as_attachment=request.args.get("download") == "1", download_name=name)


@bp.post("/history/<int:record_id>/delete")
@login_required
def delete(record_id):
    record = own_record(record_id)
    paths = [storage_path("UPLOAD_FOLDER", record.original_path), storage_path("RESULT_FOLDER", record.result_path)]
    if record.preview_path:
        paths.append(storage_path("RESULT_FOLDER", record.preview_path))
    # 先在各文件所在目录中重命名暂存；数据库失败时可恢复，避免删除一半。
    moved = []
    try:
        for path in paths:
            if path.is_file():
                temporary = path.with_name(f".delete-{uuid4().hex}-{path.name}")
                path.rename(temporary)
                moved.append((path, temporary))
        db.session.delete(record)
        db.session.commit()
    except Exception:
        db.session.rollback()
        for original, temporary in reversed(moved):
            try:
                temporary.rename(original)
            except OSError:
                current_app.logger.exception("Failed to restore file %s", original)
        current_app.logger.exception("Failed to delete record %s", record_id)
        flash("删除失败，记录已保留，请稍后再试。", "error")
        return redirect(url_for("history.detail", record_id=record_id))
    cleanup_failed = False
    for _, temporary in moved:
        try:
            temporary.unlink()
        except OSError:
            cleanup_failed = True
            current_app.logger.exception("Failed to remove staged file %s", temporary)
    current_app.logger.info("Record deleted id=%s user=%s", record_id, current_user.id)
    if cleanup_failed:
        flash("检测记录已删除，部分本地文件未能清理，请检查服务日志和存储目录权限。", "info")
    else:
        flash("检测记录及对应文件已删除。", "success")
    return redirect(url_for("history.index"))
