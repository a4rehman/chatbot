"""Tests for the [RESPONSE]/[REASONING]/[CONFIDENCE] parser."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from response_utils import parse_response


def test_full_block():
    content = "[RESPONSE]\nHello there.\n\n[REASONING]\nGreeting asked.\n\n[CONFIDENCE]\n95"
    clean, reason, conf = parse_response(content)
    assert clean == "Hello there."
    assert reason == "Greeting asked."
    assert conf == 95


def test_no_markers():
    clean, reason, conf = parse_response("Plain answer without markers.")
    assert clean == "Plain answer without markers."
    assert reason is None
    assert conf is None


def test_reasoning_without_confidence():
    content = "[RESPONSE]\nPartial.\n\n[REASONING]\nOnly reasoning."
    clean, reason, conf = parse_response(content)
    assert clean == "Partial."
    assert reason == "Only reasoning."
    assert conf is None


def test_missing_reasoning_text():
    content = "[RESPONSE]\nAnswer text.\n\n[REASONING]\n\n[CONFIDENCE]\n80"
    clean, reason, conf = parse_response(content)
    assert clean == "Answer text."
    assert reason is None
    assert conf == 80


def test_non_numeric_confidence():
    content = "[RESPONSE]\nAnswer.\n\n[REASONING]\nWhy.\n\n[CONFIDENCE]\nhigh"
    clean, reason, conf = parse_response(content)
    assert clean == "Answer."
    assert reason == "Why."
    assert conf is None


def test_confidence_with_spaces():
    content = "[RESPONSE]\nAnswer.\n\n[REASONING]\nWhy.\n\n[CONFIDENCE]\n 42 %"
    clean, reason, conf = parse_response(content)
    assert conf == 42


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL RESPONSE PARSER TESTS PASSED")