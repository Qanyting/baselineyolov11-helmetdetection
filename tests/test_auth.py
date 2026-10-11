from src.extensions import db
from src.models import User
from tests.conftest import csrf_token, login, register


def test_registration_login_remember_and_logout(app, client):
    assert register(client).status_code == 302
    with app.app_context():
        user = db.session.scalar(db.select(User))
        assert user.password_hash != "safe-password-123"
        assert user.check_password("safe-password-123")
    response = login(client)
    assert response.location == "/dashboard"
    assert "remember_token=" in " ".join(response.headers.getlist("Set-Cookie"))
    assert client.get("/dashboard").status_code == 200
    client.delete_cookie("session")
    assert client.get("/dashboard").status_code == 200
    token = csrf_token(client, "/dashboard")
    assert client.post("/logout", data={"csrf_token": token}).status_code == 302
    assert client.get("/dashboard").status_code == 302


def test_password_errors_and_duplicate_names(app, client):
    register(client)
    response = register(client, "TEST_USER")
    assert "这个用户名已被使用" in response.get_data(as_text=True)
    response = login(client, password="incorrect")
    assert "用户名或密码不正确" in response.get_data(as_text=True)
    assert client.get("/dashboard").status_code == 302


def test_csrf_required_and_external_redirect_rejected(client):
    assert client.post("/register", data={"username": "unsafe"}).status_code == 400
    register(client)
    assert login(client, next_url="//evil.example/path").location == "/dashboard"


def test_protected_pages(client):
    for path in ("/dashboard", "/detect/image", "/detect/video", "/image-detect", "/video-detect", "/history", "/history/1", "/media/1/original"):
        response = client.get(path)
        assert response.status_code == 302
        assert "/login?next=" in response.location
