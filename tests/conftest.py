import re

import pytest

from src import create_app


@pytest.fixture
def app(tmp_path):
    return create_app({"TESTING": True, "SECRET_KEY": "test-only-secret",
                       "SQLALCHEMY_DATABASE_URI": "sqlite:///" + str(tmp_path / "test.db"),
                       "UPLOAD_FOLDER": tmp_path / "uploads", "RESULT_FOLDER": tmp_path / "results",
                       "INITIALIZE_DETECTOR": False})


@pytest.fixture
def client(app):
    return app.test_client()


def csrf_token(client, path):
    response = client.get(path)
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match is not None, response.data.decode()
    return match.group(1).decode()


def register(client, username="test_user", password="safe-password-123"):
    return client.post("/register", data={"csrf_token": csrf_token(client, "/register"),
                       "username": username, "password": password, "confirm_password": password})


def login(client, username="test_user", password="safe-password-123", remember=True, next_url=""):
    return client.post("/login", data={"csrf_token": csrf_token(client, "/login"), "username": username,
                       "password": password, "remember": "on" if remember else "", "next": next_url})
