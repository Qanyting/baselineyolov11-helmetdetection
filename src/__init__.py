import logging
import os
import secrets
from datetime import timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from config import BASE_DIR, Config
from src.extensions import csrf, db, login_manager


def local_secret():
    folder = BASE_DIR / "database"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".secret_key"
    try:
        with open(path, "x", opener=lambda name, flags: os.open(name, flags, 0o600)) as file:
            file.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    return path.read_text().strip()


def create_app(test_config=None):
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    if not app.config["SECRET_KEY"]:
        app.config["SECRET_KEY"] = local_secret()
    database = make_url(app.config["SQLALCHEMY_DATABASE_URI"])
    if database.drivername == "sqlite" and database.database and database.database != ":memory:":
        Path(database.database).parent.mkdir(parents=True, exist_ok=True)
    for key in ("UPLOAD_FOLDER", "RESULT_FOLDER"):
        for kind in ("images", "videos"):
            (Path(app.config[key]) / kind).mkdir(parents=True, exist_ok=True)

    if not app.testing:
        (BASE_DIR / "logs").mkdir(exist_ok=True)
        handler = RotatingFileHandler(BASE_DIR / "logs/app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "请先登录，再进入检测平台。"
    login_manager.login_message_category = "info"

    from src.models import User
    from src.routes import auth, dashboard, detection, history
    app.register_blueprint(auth.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(detection.bp)
    app.register_blueprint(history.bp)

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (ValueError, TypeError):
            return None

    @app.template_filter("localtime")
    def localtime(value):
        if value is None:
            return "—"
        return value.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")

    @app.context_processor
    def shared_context():
        detector = app.extensions.get("helmet_detector")
        return {"model_ready": detector is not None, "model_device": getattr(detector, "device", "—"),
                "model_classes": getattr(detector, "names", {}),
                "model_metadata": getattr(detector, "metadata", {}),
                "upload_limit_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)}

    @app.errorhandler(CSRFError)
    def csrf_error(error):
        return render_template("error.html", code=400, message="页面验证已过期，请返回并刷新页面后重试。"), 400

    @app.errorhandler(413)
    def file_too_large(error):
        return render_template("error.html", code=413, message=f"文件超过 {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB，请缩小文件后重试。"), 413

    @app.errorhandler(404)
    def not_found(error):
        return render_template("error.html", code=404, message="页面或检测记录不存在。"), 404

    @app.errorhandler(SQLAlchemyError)
    @app.errorhandler(500)
    def server_error(error):
        db.session.rollback()
        app.logger.error("Application error", exc_info=error)
        return render_template("error.html", code=500, message="操作暂时未能完成，请稍后重试。"), 500

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        if not request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    with app.app_context():
        db.create_all()
    if app.config.get("INITIALIZE_DETECTOR", True):
        from src.services.detector import HelmetDetector
        from src.services.errors import ModelUnavailable
        try:
            app.extensions["helmet_detector"] = HelmetDetector(
                app.config["MODEL_PATH"], device=app.config["YOLO_DEVICE"],
                conf=app.config["YOLO_CONF"], iou=app.config["YOLO_IOU"],
                imgsz=app.config["YOLO_IMGSZ"], max_pixels=app.config["MAX_IMAGE_PIXELS"],
            )
            app.logger.info("Model ready: %s device=%s classes=%s", app.config["MODEL_PATH"],
                            app.extensions["helmet_detector"].device, app.extensions["helmet_detector"].names)
        except ModelUnavailable as error:
            app.logger.warning("%s", error)
            app.extensions["model_error"] = str(error)
    return app
