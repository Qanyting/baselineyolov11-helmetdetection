from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user, login_required
from sqlalchemy import func

from src.extensions import db
from src.models import DetectionRecord

bp = Blueprint("dashboard", __name__)


@bp.get("/")
def home():
    return redirect(url_for("dashboard.index" if current_user.is_authenticated else "auth.login"))


@bp.get("/dashboard")
@login_required
def index():
    counts = dict(db.session.execute(db.select(
        DetectionRecord.detection_type, func.count(DetectionRecord.id)
    ).where(DetectionRecord.user_id == current_user.id).group_by(DetectionRecord.detection_type)).all())
    recent = db.session.scalars(db.select(DetectionRecord).where(
        DetectionRecord.user_id == current_user.id
    ).order_by(DetectionRecord.created_at.desc(), DetectionRecord.id.desc()).limit(5)).all()
    return render_template("dashboard.html", counts=counts, recent=recent)
