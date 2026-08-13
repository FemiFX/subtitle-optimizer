"""Turn word-level ASR timestamps into readable subtitle cues.

Whisper (and friends) give you a timestamp per word. That is not a subtitle. A
subtitle has to respect a reading-speed budget, fit a line width, break where a
reader expects a break, and not flash on screen for 300ms. This library does that
conversion.

    from subtitle_optimizer import Word, optimize_transcript

    cues = optimize_transcript(words, max_cps=17, max_chars_per_line=42)

No dependencies. Grammar-aware chunking is optional — see `subtitle_optimizer.grammar`.
"""

from .converter import parse_srt_to_segments, srt_file_to_vtt_file, srt_to_vtt
from .core import (
    ChunkingPolicy,
    GrammarChunk,
    PackedCue,
    SentenceGroup,
    SubtitleCue,
    SubtitleOptimizer,
    Word,
    cues_to_segments,
    extract_cues_from_segments_json,
    extract_words_from_segments_json,
    optimize_transcript,
    rebalance_lines,
    words_from_raw_segments,
    words_from_whisper_segments,
    words_to_raw_segments,
)
from .grammar import (
    GrammarProvider,
    clear_grammar_provider,
    get_grammar_provider,
    join_texts_naturally,
    register_grammar_provider,
)

__version__ = "0.1.0"

__all__ = [
    "ChunkingPolicy",
    "GrammarChunk",
    "GrammarProvider",
    "PackedCue",
    "SentenceGroup",
    "SubtitleCue",
    "SubtitleOptimizer",
    "Word",
    "clear_grammar_provider",
    "cues_to_segments",
    "extract_cues_from_segments_json",
    "extract_words_from_segments_json",
    "get_grammar_provider",
    "join_texts_naturally",
    "optimize_transcript",
    "parse_srt_to_segments",
    "rebalance_lines",
    "register_grammar_provider",
    "srt_file_to_vtt_file",
    "srt_to_vtt",
    "words_from_raw_segments",
    "words_from_whisper_segments",
    "words_to_raw_segments",
]
