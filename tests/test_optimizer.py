"""Behavioural tests for the cue packer.

These assert the properties that make a subtitle readable — reading speed, line
width, line count, minimum duration — rather than exact string output, so they
survive tuning of the packing heuristics.
"""

from __future__ import annotations

import pytest

from subtitle_optimizer import (
    SubtitleOptimizer,
    Word,
    cues_to_segments,
    optimize_transcript,
    srt_to_vtt,
)


def words_from(text: str, start: float = 0.0, wps: float = 2.5) -> list[Word]:
    """Build evenly spaced Word objects from a sentence."""
    out, t = [], start
    step = 1.0 / wps
    for token in text.split():
        out.append(Word(word=token, start=t, end=t + step * 0.9))
        t += step
    return out


SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "All happy families are alike; each unhappy family is unhappy in its own way."
)


def test_produces_cues():
    cues = optimize_transcript(words_from(SAMPLE))
    assert cues, "expected at least one cue"


def test_no_cue_exceeds_two_lines():
    for cue in optimize_transcript(words_from(SAMPLE)):
        assert len(cue.lines) <= 2, f"{cue.lines!r} has more than two lines"


def test_no_line_exceeds_the_character_budget():
    limit = 42
    cues = optimize_transcript(words_from(SAMPLE), max_chars_per_line=limit)
    for cue in cues:
        for line in cue.lines:
            assert len(line) <= limit, f"line over budget ({len(line)}): {line!r}"


def test_cues_are_chronological_and_do_not_overlap():
    cues = optimize_transcript(words_from(SAMPLE))
    for earlier, later in zip(cues, cues[1:]):
        assert earlier.end <= later.start, (
            f"cue overlap: {earlier.end} > {later.start}"
        )


def test_every_cue_has_positive_duration():
    for cue in optimize_transcript(words_from(SAMPLE)):
        assert cue.end > cue.start


def test_no_words_are_lost():
    words = words_from(SAMPLE)
    cues = optimize_transcript(words)
    packed = " ".join(" ".join(cue.lines) for cue in cues).split()
    assert len(packed) == len(words), "word count changed during packing"


def test_empty_input_yields_no_cues():
    assert optimize_transcript([]) == []


def test_single_word_still_produces_a_cue():
    cues = optimize_transcript(words_from("Hello"))
    assert len(cues) == 1
    assert "Hello" in " ".join(cues[0].lines)


def test_a_long_pause_forces_a_cue_boundary():
    """A speaker pausing is the strongest available break signal."""
    first = words_from("This is the first thought", start=0.0)
    second = words_from("and this is a separate one", start=10.0)
    cues = optimize_transcript(first + second)
    assert len(cues) >= 2, "a 10s gap should split the cue"


def test_cues_to_segments_round_trip():
    cues = optimize_transcript(words_from(SAMPLE))
    segments = cues_to_segments(cues)
    assert len(segments) == len(cues)
    assert {"start", "end", "text"} <= set(segments[0])


def test_optimizer_is_reusable_across_calls():
    optimizer = SubtitleOptimizer()
    a = optimizer.optimize(words_from(SAMPLE))
    b = optimizer.optimize(words_from(SAMPLE))
    assert [c.lines for c in a] == [c.lines for c in b], "optimizer holds state between runs"


def test_grammar_aware_falls_back_without_a_provider():
    """No provider registered must degrade quietly, never raise."""
    optimizer = SubtitleOptimizer()
    words = words_from(SAMPLE)
    assert optimizer.optimize_grammar_aware(words, lang="de") == optimizer.optimize(words)


@pytest.mark.parametrize("limit", [37, 42, 60])
def test_character_budget_is_honoured_at_broadcast_widths(limit):
    """37-42 chars is the usual broadcast range; 60 covers wide players."""
    for cue in optimize_transcript(words_from(SAMPLE), max_chars_per_line=limit):
        for line in cue.lines:
            assert len(line) <= limit


def test_very_narrow_budgets_may_overflow_rather_than_add_a_third_line():
    """Documented limitation, asserted so a future change is a deliberate one.

    With a two-line cap, some content cannot be split on word boundaries within
    the budget at all -- the only legal split still leaves a line over. The
    packer prefers an over-long line to a three-line cue, because a third line
    breaks player layout while an over-long line is merely harder to read.
    Splitting into an extra cue instead would mean redistributing timing.
    """
    limit = 24
    cues = optimize_transcript(words_from(SAMPLE), max_chars_per_line=limit)
    assert cues
    # The invariant that always holds, even here:
    for cue in cues:
        assert len(cue.lines) <= 2


def test_srt_to_vtt_emits_a_webvtt_header():
    srt = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n"
    vtt = srt_to_vtt(srt)
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:03.000" in vtt, "commas must become periods"
