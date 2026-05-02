from __future__ import annotations

from bitforge.esp.model import EspTaskExample, TinyEspController


def test_tiny_esp_controller_predicts_action():
    model = TinyEspController()
    report = model.train_from_defaults()
    assert report.examples_seen > 0
    pred = model.predict("turn on the light now")
    assert pred.label in {"control_light", "status_check", "set_mode", "help", "idle", "set_timer", "launch_script", "play_media", "read_sensor", "connect_wifi", "reboot_device", "set_temperature"}
    assert pred.actions
    assert pred.tokens_per_second_estimate > 0


def test_export_profile_shrinks():
    model = TinyEspController()
    model.train_from_defaults()
    profile = model.export_tiny_profile()
    assert profile["compression_ratio"] > 1.0
    assert profile["estimated_profile_bytes"] > 0


def test_custom_training_changes_result():
    model = TinyEspController()
    model.fit([
        EspTaskExample("turn on the light", "control_light"),
        EspTaskExample("what is the status", "status_check"),
        EspTaskExample("set performance mode", "set_mode"),
        EspTaskExample("please reboot now", "reboot_device"),
    ])
    pred = model.predict("please reboot the device")
    assert pred.label == "reboot_device"
