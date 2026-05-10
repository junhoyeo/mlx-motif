"""Tests for the ThinkFilter (server-side <think>...</think> stream filter)."""

from __future__ import annotations

import pytest

from mlx_motif.server import ThinkFilter


@pytest.mark.parametrize("mode", ["visible", "hidden", "captured"])
def test_no_think_tag_passthrough(mode):
    f = ThinkFilter(mode)
    out = f.feed("Hello, world!")
    assert out == "Hello, world!"
    assert f.captured == ""


def test_visible_mode_passes_tags_through():
    f = ThinkFilter("visible")
    parts = ["The ", "<think>", "let me ", "think</think>", " answer is 4."]
    out = "".join(f.feed(p) for p in parts)
    assert out == "The <think>let me think</think> answer is 4."


def test_hidden_mode_drops_think_block():
    f = ThinkFilter("hidden")
    parts = ["The ", "<think>", "let me ", "think</think>", " answer is 4."]
    out = "".join(f.feed(p) for p in parts)
    assert out == "The  answer is 4."
    assert f.captured == ""


def test_captured_mode_collects_trace():
    f = ThinkFilter("captured")
    parts = ["The ", "<think>", "let me ", "think</think>", " answer is 4."]
    out = "".join(f.feed(p) for p in parts)
    assert out == "The  answer is 4."
    assert f.captured == "let me think"


def test_partial_tag_split_across_chunks():
    """The <think> tag is split mid-tag across two stream chunks."""
    f = ThinkFilter("hidden")
    out1 = f.feed("Hello <thi")
    out2 = f.feed("nk>secret</think> world")
    assert out1 + out2 == "Hello  world"


def test_partial_close_tag_split():
    f = ThinkFilter("hidden")
    out1 = f.feed("a<think>b</thin")
    out2 = f.feed("k>c")
    assert out1 + out2 == "ac"


def test_unclosed_think_block_in_hidden_mode():
    """If the stream ends mid-think, hidden mode emits the prefix only."""
    f = ThinkFilter("hidden")
    out = f.feed("prefix<think>not closed yet")
    assert out == "prefix"


def test_multiple_think_blocks():
    f = ThinkFilter("captured")
    out = f.feed("a<think>x</think>b<think>y</think>c")
    assert out == "abc"
    assert f.captured == "xy"
