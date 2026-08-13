"""Optional grammar-aware chunking, supplied by the host application.

The core cue packer needs no linguistic analysis: it breaks on punctuation,
pauses and character budgets. That path has no dependencies and is what most
callers want.

A grammar-aware path also exists. It splits text at *syntactic* boundaries
rather than merely at commas, which produces noticeably better line breaks for
languages with long verb-final clauses — German especially, where breaking
before a separable prefix or a clause-final verb reads badly.

That path needs a sentence parser (spaCy or similar) and a chunker, which are
heavy and language-specific, so this library does not ship them. Instead, register
your own:

    from subtitle_optimizer import register_grammar_provider

    register_grammar_provider(
        parse_sentence=my_parse_sentence,
        is_parser_available=lambda lang: lang in {"de", "en"},
        chunk_sentence=my_chunk_sentence,
        join_texts_naturally=my_join,
        finalize_cue=my_finalize_cue,
        enforce_min_duration=my_enforce_min_duration,
        enforce_cue_monotonic_timing=my_enforce_monotonic,
    )

With no provider registered, `SubtitleOptimizer.optimize_grammar_aware()` logs
once and falls back to `optimize()`. Nothing raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class GrammarProvider:
    """Callables the grammar-aware path needs. All are optional."""

    parse_sentence: Optional[Callable[..., Any]] = None
    is_parser_available: Optional[Callable[[Optional[str]], bool]] = None
    chunk_sentence: Optional[Callable[..., Any]] = None
    join_texts_naturally: Optional[Callable[[list], str]] = None
    finalize_cue: Optional[Callable[..., Any]] = None
    enforce_min_duration: Optional[Callable[..., Any]] = None
    enforce_cue_monotonic_timing: Optional[Callable[..., Any]] = None

    @property
    def can_parse(self) -> bool:
        return callable(self.parse_sentence) and callable(self.chunk_sentence)


_provider = GrammarProvider()


def register_grammar_provider(**kwargs) -> GrammarProvider:
    """Register grammar callables. Returns the active provider."""
    global _provider
    unknown = set(kwargs) - set(GrammarProvider.__dataclass_fields__)
    if unknown:
        raise TypeError(f"Unknown grammar provider hooks: {sorted(unknown)}")
    _provider = GrammarProvider(**kwargs)
    return _provider


def get_grammar_provider() -> GrammarProvider:
    return _provider


def clear_grammar_provider() -> None:
    """Reset to the no-provider state. Mainly for tests."""
    global _provider
    _provider = GrammarProvider()


def join_texts_naturally(texts: list) -> str:
    """Join chunk texts, using the provider's joiner when one is registered.

    The built-in fallback is space-joining with a guard against a leading
    punctuation mark or contraction becoming its own token — otherwise text
    reads as "... it 's ..." rather than "... it's ...".
    """
    if callable(_provider.join_texts_naturally):
        return _provider.join_texts_naturally(texts)

    parts = [t for t in (texts or []) if t]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        if part[:1] in ",.;:!?)]}%" or part.startswith("'") or part.startswith("’"):
            out += part
        elif out.endswith(("(", "[", "{", "-", "–", "—")):
            out += part
        else:
            out += " " + part
    return out
