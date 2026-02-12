"""Offline test suite for environments without Flask/network package access.

This script injects a tiny Flask stub so `app.py` can be imported and core
logic can be tested without installing external packages.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def inject_flask_stub() -> None:
    flask_stub = types.ModuleType("flask")

    class FlaskStub:
        def __init__(self, *_args, **_kwargs):
            pass

        def route(self, *_args, **_kwargs):
            def deco(func):
                return func

            return deco

    flask_stub.Flask = FlaskStub
    flask_stub.render_template = lambda *_a, **_k: "rendered"
    flask_stub.request = types.SimpleNamespace(form={})
    sys.modules["flask"] = flask_stub


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_training_artifact() -> None:
    run_command([sys.executable, "-m", "py_compile", "app.py", "train_model.py"])
    run_command([sys.executable, "train_model.py"])

    model_path = Path("model/news_nb_model.json")
    assert_true(model_path.exists(), "Model artifact was not created")
    payload = json.loads(model_path.read_text(encoding="utf-8"))

    assert_true(payload.get("model_type") == "multinomial_naive_bayes", "Unexpected model type")
    assert_true("validation" in payload and payload["validation"].get("f1", 0) > 0.6, "Validation F1 too low")
    assert_true(len(payload.get("token_log_prob", {})) > 1000, "Model vocabulary unexpectedly small")


def test_app_core_logic_without_flask_install() -> None:
    inject_flask_stub()
    import app  # noqa: PLC0415

    text, extracted = app._resolve_input_text("Some manual claim text", "")
    assert_true(text == "Some manual claim text", "Manual text path failed")
    assert_true(extracted == "", "Manual text should not set extracted text")

    try:
        app._validate_public_url("http://127.0.0.1/test")
    except ValueError:
        pass
    else:
        raise AssertionError("Private URL should be rejected")

    app._fetch_tweet_text = lambda _u: "tweet extracted text"
    text2, extracted2 = app._resolve_input_text("", "https://x.com/user/status/123456")
    assert_true(text2 == "tweet extracted text", "Tweet URL extraction path failed")
    assert_true(extracted2 == "tweet extracted text", "Tweet extracted text not returned")


if __name__ == "__main__":
    test_training_artifact()
    test_app_core_logic_without_flask_install()
    print("offline tests passed")
