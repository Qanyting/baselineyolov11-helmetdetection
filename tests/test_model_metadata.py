import hashlib
import json

import pytest

from src.services.model_metadata import read_model_metadata
from tests.conftest import login, register


def test_metadata_is_bound_to_exact_weights(tmp_path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"model A")
    model.with_suffix(".json").write_text(json.dumps({
        "sha256": hashlib.sha256(b"model A").hexdigest(), "notice": "Cycling dataset",
        "training_domain": "Cycling helmets", "unused": "discard this key",
    }))
    assert read_model_metadata(model) == {"notice": "Cycling dataset", "training_domain": "Cycling helmets"}
    model.write_bytes(b"model B")
    assert read_model_metadata(model) == {}


@pytest.mark.parametrize("sidecar", [None, "broken JSON", "[]"])
def test_missing_or_invalid_metadata_does_not_block_model(tmp_path, sidecar):
    model = tmp_path / "best.pt"
    model.write_bytes(b"weights")
    if sidecar is not None:
        model.with_suffix(".json").write_text(sidecar)
    assert read_model_metadata(model) == {}


def test_app_displays_real_model_limitations_and_threshold_help(app, client):
    class Detector:
        names = {0: "helmet", 1: "no_helmet"}
        device = "cpu"
        metadata = {"notice": "骑行头盔数据训练，工地场景存在漏检。", "training_domain": "骑行头盔"}
    app.extensions["helmet_detector"] = Detector()
    register(client)
    login(client)
    page = client.get("/detect/image").get_data(as_text=True)
    assert "工地场景存在漏检" in page
    assert "这是预测分数门槛，不是模型准确率" in page
    assert "恢复 0.25" in page
    assert "模型已加载" in page
