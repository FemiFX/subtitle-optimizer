# subtitle-optimizer

Turn **word-level ASR timestamps into readable subtitle cues.**

Whisper and its peers hand you a timestamp per word. That isn't a subtitle. A
subtitle has to respect a reading-speed budget, fit a line width, break where a
reader expects a break, and not flash on screen for 300 milliseconds. This
library does that conversion.

```python
from subtitle_optimizer import optimize_transcript, words_from_whisper_segments

words = words_from_whisper_segments(result["segments"])
cues = optimize_transcript(words, max_chars_per_line=42, max_cps=21)

for cue in cues:
    print(f"{cue.start:.2f} --> {cue.end:.2f}  ({cue.cps:.1f} cps)")
    print(cue.text)
```

**No dependencies.** Standard library only.

## Why not just split on punctuation

Because the constraints interact. Split purely on commas and you get cues that
sit on screen for 400ms, or 90-character lines, or a cue that reads at 40
characters per second. Split purely on character count and you break mid-clause.
Split purely on timing and you break mid-word.

The packer weighs all of them at once:

| Constraint | Default | Why |
|---|---|---|
| Characters per second | target 18, max 21 | Reading speed. Above ~21 cps most viewers can't keep up |
| Characters per line | 42 | Broadcast convention is 37–42 |
| Lines per cue | 2 | A third line covers the picture and breaks player layout |
| Minimum duration | 1.0s | Below this a cue reads as a flash |
| Maximum duration | 7.0s | Beyond this the cue outstays its audio |
| Pause threshold | 0.3s / 0.7s | A speaker pausing is the strongest available break signal |

## What it does beyond packing

**Prefers linguistically sound breaks.** A line shouldn't end on a dangling
function word — *"So a part of my / research is…"* reads worse than a lopsided
split. Balance is only the tiebreaker; punctuation gets a bonus, a trailing
function word gets a penalty. Function-word lists are per-language.

**Extends over-budget cues into following silence.** The standard remediation
for a cue that reads too fast is to give the viewer the missing time, taken from
the gap before the next cue rather than by cutting words.

**Folds stranded short cues.** A lone `"So"` or an orphaned `"things."` gets
merged back into an adjacent cue when the merge doesn't break another rule.

**Carries speaker labels through.** When the ASR provider returns diarization,
each cue exposes its dominant speaker, so a cue spanning a speaker change is
still labelled sensibly.

**Keeps filler words.** `"um"`, `"uh"` are flagged rather than dropped, so the
caller decides whether a verbatim transcript or a clean read is wanted.

## Grammar-aware mode (optional)

Punctuation is a decent proxy for clause structure in English. It is a poor one
in German, where a clause-final verb or a separable prefix routinely lands where
no comma does — so a punctuation-driven break splits the verb from its subject.

`optimize_grammar_aware()` breaks at syntactic boundaries instead. It needs a
sentence parser and chunker, which are heavy and language-specific, so this
library doesn't ship them. Register your own:

```python
from subtitle_optimizer import register_grammar_provider

register_grammar_provider(
    parse_sentence=my_parser,
    is_parser_available=lambda lang: lang in {"de", "en"},
    chunk_sentence=my_chunker,
)
```

With no provider registered it logs once and falls back to the dependency-free
packer. It never raises.

## Install

```bash
pip install subtitle-optimizer
```

## Output

```python
cue.start        # float seconds
cue.end
cue.lines        # list[str], at most max_lines_per_cue
cue.text         # lines joined with newline
cue.duration
cue.cps          # characters per second, the readability metric
cue.words        # the Word objects behind this cue
cue.to_dict()    # JSON-ready, includes speaker when diarized
```

Helpers for moving between formats:

```python
words_from_whisper_segments(segments)     # Whisper output -> Word list
words_from_raw_segments(raw_words)        # generic {word,start,end} dicts
cues_to_segments(cues)                    # cues -> {start,end,text} dicts
rebalance_lines(cues, max_chars_per_line) # re-wrap after editing or translating
srt_to_vtt(srt_text)                      # SRT -> WebVTT
parse_srt_to_segments(srt_text)           # SRT -> segment dicts
```

## Known limitation

At **very narrow line budgets** (roughly under 30 characters), a cue whose
content cannot be split on word boundaries within `max_chars_per_line × 2` will
emit one over-long line rather than a third line. A third line breaks player
layout; an over-long line is merely harder to read. Splitting into an extra cue
instead would require redistributing timing, which the packer deliberately
doesn't do at this stage. This is asserted in the test suite so any future change
is a deliberate one.

At broadcast widths (37–42) the budget is always honoured.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

17 tests. They assert the properties that make a subtitle readable — reading
speed, line width, line count, chronological non-overlapping cues, no words lost
— rather than exact strings, so they survive tuning of the heuristics.

## Provenance

Extracted from the media pipeline of a production content platform, where it
processes ASR output for multi-language subtitle delivery.

## Licence

MIT — see [LICENSE](LICENSE).
