from __future__ import annotations

from bitforge.esp.model import EspTaskExample, TinyEspController


def test_tiny_esp_controller_predicts_action():
    model = TinyEspController()
    model.fit([
        EspTaskExample(text="turn on the light", label="control_light"),
        EspTaskExample(text="check status", label="status_check"),
        EspTaskExample(text="fast mode please", label="set_mode"),
    ])
    pred = model.predict("turn on the light now")
    assert pred.label in {"control_light", "status_check", "set_mode", "help", "idle"}
    assert pred.actions
    assert pred.tokens_per_second_estimate > 0


def test_export_profile_shrinks():
    model = TinyEspController()
    profile = model.export_tiny_profile()
    assert profile["compression_ratio"] > 1.0
