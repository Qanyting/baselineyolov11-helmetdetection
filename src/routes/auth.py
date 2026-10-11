import re
import unicodedata
from urllib.parse import urlsplit

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError

from src.extensions import db
from src.models import User
from src.models.user import utc_now

bp = Blueprint("auth", __name__)


def normalize_username(value):
    return unicodedata.normalize("NFKC", value).strip()


def safe_next(value):
    # 仅允许本站绝对路径，拒绝协议相对 URL、反斜杠和控制字符。
    if not value or not value.startswith("/") or value.startswith("//"):
        return None
    if "\\" in value or any(ord(char) < 32 for char in value):
        return None
    parts = urlsplit(value)
    return value if not parts.netloc and not parts.scheme else None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    username = ""
    if request.method == "POST":
        username = normalize_username(request.form.get("username", ""))
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not re.fullmatch(r"[\w\-]{3,32}", username):
            flash("用户名需为 3–32 个中文、字母、数字、下划线或短横线。", "error")
        elif not 8 <= len(password) <= 128:
            flash("密码长度需为 8–128 个字符。", "error")
        elif password != confirm:
            flash("两次输入的密码不一致。", "error")
        else:
            user = User(username=username, username_key=username.casefold())
            user.set_password(password)
            db.session.add(user)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash("这个用户名已被使用，请换一个。", "error")
            else:
                current_app.logger.info("User registered id=%s", user.id)
                flash("账户已创建，请登录。", "success")
                return redirect(url_for("auth.login"))
    return render_template("register.html", username=username)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    username = ""
    next_url = safe_next(request.values.get("next", "")) or ""
    if request.method == "POST":
        username = normalize_username(request.form.get("username", ""))
        password = request.form.get("password", "")
        user = db.session.scalar(db.select(User).where(User.username_key == username.casefold()))
        if user is None or len(password) > 128 or not user.check_password(password):
            flash("用户名或密码不正确。", "error")
        else:
            user.last_login = utc_now()
            db.session.commit()
            login_user(user, remember=request.form.get("remember") == "on")
            current_app.logger.info("User logged in id=%s", user.id)
            return redirect(next_url or url_for("dashboard.index"))
    return render_template("login.html", username=username, next_url=next_url)


@bp.post("/logout")
@login_required
def logout():
    current_app.logger.info("User logged out id=%s", current_user.id)
    logout_user()
    flash("你已退出登录。", "success")
    return redirect(url_for("auth.login"))
