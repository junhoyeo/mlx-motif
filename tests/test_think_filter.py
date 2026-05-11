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


def test_start_in_think_hidden_drops_pre_close_text():
    """When the chat template puts the model already in <think> mode
    (Motif's reasoning template does this), the stream starts INSIDE
    the block — only a closing </think> exists in the response, no
    opening tag. `start_in_think=True` handles this."""
    f = ThinkFilter("hidden", start_in_think=True)
    out = f.feed("reasoning here</think>answer here")
    assert out == "answer here"


def test_start_in_think_captured_collects_pre_close():
    f = ThinkFilter("captured", start_in_think=True)
    out = f.feed("reasoning here</think>answer here")
    assert out == "answer here"
    assert f.captured == "reasoning here"


def test_start_in_think_visible_still_passthrough():
    f = ThinkFilter("visible", start_in_think=True)
    out = f.feed("reasoning</think>answer")
    assert out == "reasoning</think>answer"


def test_start_in_think_with_chunk_split_close_tag():
    """Close tag split across stream chunks while starting in-think."""
    f = ThinkFilter("hidden", start_in_think=True)
    out1 = f.feed("reasoning</thi")
    out2 = f.feed("nk>answer")
    assert out1 + out2 == "answer"


# --------------------------------------------------------------------------- #
# Prompt-tail detection (_prompt_opens_think_block)
# --------------------------------------------------------------------------- #


def test_prompt_opens_think_block_motif_template():
    """Motif's reasoning template ends with `<|assistant|><think>\\n`."""
    from mlx_motif.server import _prompt_opens_think_block

    p = "<|system|>...<|user|>hi<|endofturn|><|assistant|><think>\n"
    assert _prompt_opens_think_block(p) is True


def test_prompt_opens_think_block_no_think_tag():
    from mlx_motif.server import _prompt_opens_think_block

    p = "<|system|>...<|user|>hi<|endofturn|><|assistant|>"
    assert _prompt_opens_think_block(p) is False


def test_prompt_opens_think_block_completed_prior_turn():
    """A complete prior <think>...</think> followed by a fresh assistant
    turn must not trigger — the next turn is not in think mode."""
    from mlx_motif.server import _prompt_opens_think_block

    p = "<|assistant|><think>prev reasoning</think>prev answer<|endofturn|><|assistant|>"
    assert _prompt_opens_think_block(p) is False


def test_prompt_opens_think_block_user_literal_think_text():
    """REGRESSION GUARD: a user message containing the literal text
    `<think>` (e.g., asking about the tag) must NOT trigger detection —
    the assistant turn after it does NOT start in think mode.
    Caught by codex review of PR #4."""
    from mlx_motif.server import _prompt_opens_think_block

    p = "<|user|>what does <think> mean?<|endofturn|><|assistant|>"
    assert _prompt_opens_think_block(p) is False


def test_prompt_opens_think_block_trailing_whitespace():
    """Trailing whitespace after the open tag must still count as open."""
    from mlx_motif.server import _prompt_opens_think_block

    assert _prompt_opens_think_block("<|assistant|><think>") is True
    assert _prompt_opens_think_block("<|assistant|><think>\n") is True
    assert _prompt_opens_think_block("<|assistant|><think>\n\n\t  ") is True
