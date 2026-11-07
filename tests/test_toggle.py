"""The /vcc switch: its state file and the command text. No Hermes needed."""

from __future__ import annotations

import pytest

import toggle

LOCAL_ONLY = {"context": {"vcc": {"only_local_models": True}}}


@pytest.fixture(autouse=True)
def state_file(monkeypatch, tmp_path):
    path = tmp_path / toggle.STATE_FILE
    monkeypatch.setattr(toggle, "state_path", lambda: path)
    toggle._cache.update(key=None, value=None)
    return path


@pytest.fixture()
def config(monkeypatch):
    cfg: dict = {}
    monkeypatch.setattr(toggle, "load_config", lambda: cfg)
    return cfg


def test_no_file_means_the_config_decides(state_file):
    assert not state_file.exists() and toggle.hosted_override() is None
    assert toggle.hosted_mode({}) == (True, "config")
    assert toggle.hosted_mode(LOCAL_ONLY) == (False, "config")


@pytest.mark.parametrize("value", [True, False])
def test_switch_round_trip(state_file, value):
    toggle.set_hosted_override(value)
    assert toggle.hosted_override() is value
    assert toggle.hosted_mode(LOCAL_ONLY) == (value, "/vcc")
    toggle.set_hosted_override(None)
    assert not state_file.exists() and toggle.hosted_override() is None


def test_flipping_is_seen_at_once(state_file):
    for value in (True, False, True, False):
        toggle.set_hosted_override(value)
        assert toggle.hosted_override() is value


@pytest.mark.parametrize("junk", ["", "{", "[]", '{"hosted_models": "yes"}', '{"other": true}'])
def test_an_unreadable_file_leaves_it_to_the_config(state_file, junk):
    state_file.write_text(junk)
    assert toggle.hosted_override() is None


def test_write_is_atomic_and_leaves_no_temp_files(state_file):
    toggle.set_hosted_override(True)
    assert [p.name for p in state_file.parent.iterdir()] == [toggle.STATE_FILE]
    assert state_file.stat().st_mode & 0o777 == 0o600


def test_status_without_arguments_changes_nothing(state_file, config):
    config.update(LOCAL_ONLY)
    text = toggle.handle("")
    assert "hosted models" in text and ": OFF (from config)" in text and "only_local_models: true" in text
    assert not state_file.exists()


@pytest.mark.parametrize("arg,value,word", [("on", True, "ON"), ("OFF", False, "OFF"), (" yes ", True, "ON")])
def test_on_and_off(state_file, config, arg, value, word):
    text = toggle.handle(arg)
    assert toggle.hosted_override() is value and f": {word} (set with /vcc)" in text


def test_default_clears_the_switch(state_file, config):
    config.update(LOCAL_ONLY)
    toggle.handle("on")
    text = toggle.handle("default")
    assert not state_file.exists() and ": OFF (from config)" in text


def test_unknown_argument_shows_usage_and_changes_nothing(state_file, config):
    toggle.handle("on")
    text = toggle.handle("sideways")
    assert text.startswith("Usage: /vcc") and toggle.hosted_override() is True


def test_hermes_session_keywords_are_accepted(config):
    # Hermes passes the session context to handlers that declare it; **kwargs takes it all.
    text = toggle.handle("off", source=object(), session_id="s", session_key="k", platform="webui")
    assert ": OFF (set with /vcc)" in text


def test_a_failed_save_says_so(monkeypatch, config):
    def boom(value):
        raise PermissionError("read-only")

    monkeypatch.setattr(toggle, "set_hosted_override", boom)
    assert "could not save" in toggle.handle("on") and toggle.hosted_override() is None
