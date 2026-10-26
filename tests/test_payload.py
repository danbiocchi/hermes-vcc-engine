"""Tests for bench/payload.py using made-up messages only."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import payload  # noqa: E402

from transcripts import untrusted  # noqa: E402


def turn(call_id, name, args, content):
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": call_id, "content": content}]


def test_facts_that_only_a_result_holds():
    turns = [*turn("a", "search_files", {"pattern": "*idea*"}, json.dumps({"total_count": 2, "files": ["x"]})),
             *turn("b", "read_file", {"path": "/w/ideas.md"}, json.dumps({"content": "1|x", "total_lines": 12})),
             *turn("c", "web_search", {"query": "llama ui"}, untrusted("web_search", {
                 "success": True, "data": {"web": [{"url": "https://www.fedoramagazine.org/x"}, {"url": "https://github.com/y"}]}})),
             *turn("d", "clarify", {}, json.dumps({"responses": [
                 {"question": "Which email account?", "choices_offered": ["New dedicated Gmail", "Your Hotmail"],
                  "user_response": "New dedicated Gmail"},
                 {"question": "Watch HN?", "user_response": None}]})),
             *turn("e", "read_file", {"path": "/w/gone.md"}, json.dumps({"error": "not found", "total_lines": 0}))]
    probes = {p["family"]: p for p in payload.make_probes(turns)}
    assert set(probes) == {"search_count", "file_lines", "web_domain", "clarify_answer"}
    assert probes["search_count"]["expect"] == 2 and "`*idea*`" in probes["search_count"]["q"]
    assert probes["file_lines"]["expect"] == 12 and "`ideas.md`" in probes["file_lines"]["q"]
    assert probes["web_domain"]["expect"] == ["fedoramagazine.org", "github.com"]
    assert probes["clarify_answer"]["expect"] == ["dedicated", "gmail"]  # the words other choices don't share


def test_grade():
    assert payload.grade({"grade": "number", "expect": 1204}, "It has 1,204 lines.")
    assert not payload.grade({"grade": "number", "expect": 12}, "120 lines")
    assert payload.grade({"grade": "any_of", "expect": ["github.com"]}, "GitHub.com")
    assert payload.grade({"grade": "any_word", "expect": ["gmail"]}, "They chose the Gmail one.")
    assert not payload.grade({"grade": "any_word", "expect": ["gmail"]}, "Hotmail")


def test_paraphrase_changes_the_wording():
    for p in payload.make_probes(turn("a", "search_files", {"pattern": "x"}, json.dumps({"total_count": 3}))):
        assert payload.paraphrase(p)["q"] != p["q"] and payload.paraphrase(p)["expect"] == p["expect"]
