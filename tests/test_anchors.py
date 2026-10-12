"""anchors.build() must equal Hermes's _build_anchor_index() byte for byte, always."""

from __future__ import annotations

import random
import re

import pytest

from conftest import requires_hermes
from transcripts import fix_the_build, long_session

pytestmark = requires_hermes


@pytest.fixture()
def cc():
    import agent.context_compressor as module

    return module


@pytest.fixture(autouse=True)
def _fresh_footer():
    import anchors

    anchors._FOOTER = None
    yield
    anchors._FOOTER = None


def same(cc, turns):
    import anchors

    return anchors.build(turns, cc) == cc._build_anchor_index(turns)


_BITS = [
    "src/app.py", "./a/b.json", "docs/README.md", "x.py", "a/b.tsx", "c/d.sh.", "e/f.yml,", "fix/login", "feat/ui-x",
    "ent/thing", "salvage/old", "prefix/fix/x", "ValueError: boom", "Exception", "ENOSPC", "EACCES x", "SIGKILL",
    "Traceback (most recent call last)", "KeyError", "#1234", "#12", "#123456", "@someone", "@teknium", "@ab",
    "https://example.com/a/very/long/path", "http://short.io/abcdefghij)", "deadbeefcafe", "0123456789abcdef01",
    "DEADBEEF01", "xdeadbeef012", "é", "中文", "🙂", "\u2028", "\r\n", "\n", " ", ".", ",", ";", ":", "_", "-",
]


def test_verify_passes_against_this_hermes(cc):
    import anchors

    assert anchors.verify(cc) is True


def test_identical_on_fixtures_and_transcripts(cc):
    import anchors

    assert same(cc, anchors._fixture())
    assert same(cc, [])
    assert same(cc, [{"content": None}, {"content": ""}])
    assert same(cc, fix_the_build())
    for seed in range(3):
        assert same(cc, long_session(150, output_chars=2000, seed=seed))


def test_identical_on_random_turn_lists(cc):
    rng = random.Random(5)
    for _ in range(3000):
        turns = [{"content": "".join(rng.choice(_BITS) for _ in range(rng.randint(0, 40)))}
                 for _ in range(rng.randint(1, 8))]
        assert same(cc, turns), turns


def test_identical_on_sparse_dense_and_huge_text(cc):
    rng = random.Random(8)
    filler = "plain log line with nothing special in it\n" * 2000
    sparse = filler + "error in src/main.py: ValueError: nope\n" + filler + "see https://example.com/issue/77 #4321\n"
    dense = "\n".join(" ".join(rng.choice(_BITS) for _ in range(6)) for _ in range(5000))
    huge = "\n".join(" ".join(rng.choice(_BITS + ["word"] * 30) for _ in range(12)) for _ in range(40_000))
    for text in (sparse, dense, huge, "no newline src/x.py fix/abc deadbeefcafe", "trailing\n", "\n\nsrc/a.py\n\n"):
        assert same(cc, [{"content": text}])


def test_identical_with_crlf_unicode_and_lone_surrogates(cc):
    text = "a\r\nsrc/x.py\r\n\ud800deadbeefcafe0\udfff\r\néé fix/abc-def\r\n" * 50
    assert same(cc, [{"content": text}, {"content": "中文 ValueError: x\r\n"}])


def test_unknown_pattern_is_scanned_in_full(cc, monkeypatch):
    extra = ("tickets", re.compile(r"JIRA-\d+"), 10)
    monkeypatch.setattr(cc, "_ANCHOR_PATTERNS", [*cc._ANCHOR_PATTERNS, extra])
    assert same(cc, [{"content": "fix JIRA-12 in src/a.py\nJIRA-12 again, JIRA-9"}])


def test_changed_pattern_falls_back_to_a_full_scan(cc, monkeypatch):
    patched = [(label, re.compile(pat.pattern + "|zzz-anchor") if label == "files" else pat, cap)
               for label, pat, cap in cc._ANCHOR_PATTERNS]
    monkeypatch.setattr(cc, "_ANCHOR_PATTERNS", patched)
    assert same(cc, [{"content": "zzz-anchor here\nsrc/a.py there"}])


def test_verify_catches_a_broken_prefilter(cc, monkeypatch):
    import anchors

    files_src = next(p.pattern for label, p, _ in cc._ANCHOR_PATTERNS if label == "files")
    monkeypatch.setitem(anchors._PREFILTERS, files_src, ("literals", (".nothing",), "/"))
    assert anchors.verify(cc) is False


def test_prefilters_only_cover_patterns_hermes_has(cc):
    import anchors

    assert set(anchors._PREFILTERS) <= {p.pattern for _, p, _ in cc._ANCHOR_PATTERNS}
