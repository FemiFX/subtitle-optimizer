"""
Subtitle optimization algorithm using word-level timestamps.

Converts raw word-level transcript data from Whisper into optimized
subtitle cues based on:
- Characters per second (CPS) limits
- Maximum characters per line
- Two lines per subtitle block
- Pause detection for natural breaks
- Punctuation-aware line breaking
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .grammar import get_grammar_provider, join_texts_naturally


@dataclass
class Word:
    """A single word with timing information."""

    word: str
    start: float
    end: float
    probability: float = 1.0
    # Optional speaker label, populated only when the ASR provider returned
    # diarization (currently AssemblyAI). None for in-house Whisper / Groq.
    # Serializers forward this field through to segments_json when non-None.
    speaker: Optional[str] = None
    # Disfluency flag ("um"/"uh"/…). Filler words are KEPT in the persisted
    # word stream (so reprocess is lossless and the acoustic span they occupy
    # stays visible as an inter-word pause) but excluded from cue/segment/
    # transcript text by the assembly step.
    is_filler: bool = False

    @property
    def duration(self) -> float:
        return max(0, self.end - self.start)

    @property
    def char_count(self) -> int:
        return len(self.word.strip())


@dataclass
class SubtitleCue:
    """A single subtitle cue (block) with one or two lines."""

    start: float
    end: float
    lines: List[str]
    words: List[Word]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def duration(self) -> float:
        return max(0.001, self.end - self.start)

    @property
    def char_count(self) -> int:
        return sum(len(line) for line in self.lines)

    @property
    def cps(self) -> float:
        """Characters per second."""
        return self.char_count / self.duration

    def to_dict(self) -> dict:
        d = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "lines": self.lines,
            "cps": round(self.cps, 2),
        }
        # When the underlying words carry speaker labels (provider was
        # AssemblyAI / diarized), expose the dominant speaker on the cue
        # for SRT display and the transcript editor. Cues that span a
        # speaker change get the most-frequent label.
        speakers = [w.speaker for w in self.words if w.speaker is not None]
        if speakers:
            d["speaker"] = max(set(speakers), key=speakers.count)
        return d


@dataclass
class SentenceGroup:
    """A complete sentence spanning one or more cues for translation."""

    sentence_text: str  # Complete sentence text
    source_cue_indices: List[int]  # Indices of source cues that form this sentence
    start_time: float  # Start time (from first cue)
    end_time: float  # End time (from last cue)
    is_complete: bool = True  # Whether sentence ends with terminal punctuation


# =============================================================================
# Grammar-Aware Translation Dataclasses
# =============================================================================


@dataclass
class AlignedToken:
    """A target language token with alignment and parse info."""

    idx: int  # Position in target token list
    text: str  # Token text
    char_start: int  # Character offset start in target sentence
    char_end: int  # Character offset end in target sentence
    aligned_source_indices: List[int]  # Source word indices this aligns to
    pos: str  # Universal POS tag (VERB, NOUN, ADP, etc.)
    dep: str  # Dependency relation (nsubj, obj, aux, etc.)
    head_idx: int  # Index of syntactic head (-1 for root)
    is_aligned: bool = True  # False if no alignment found


@dataclass
class ParsedSentence:
    """Complete parse output for a translated sentence."""

    text: str  # Original sentence text
    tokens: List[AlignedToken]  # Tokens with parse + alignment info
    lang: str  # Language code
    parse_model: str  # spaCy model used (e.g., "de_core_news_sm")


@dataclass
class GrammarChunk:
    """A grammatically safe unit of target text."""

    chunk_id: int
    token_indices: List[int]  # Indices into ParsedSentence.tokens
    text: str  # Chunk text (joined tokens)
    head_token_idx: int  # Primary head of this chunk
    start_time: Optional[float]  # Projected start time (from alignment)
    end_time: Optional[float]  # Projected end time (from alignment)
    anchor_confidence: float  # 0-1, how confident is timing projection
    char_count: int  # Character count for CPS calculation
    first_pos: Optional[str] = None  # POS tag of first token in chunk
    last_pos: Optional[str] = None  # POS tag of last token in chunk
    # Chunker break score for the boundary BEFORE this chunk (lower = better
    # place to break; 0 for the first chunk of a sentence). Carried through so
    # the cue packer can rank candidate cue boundaries by linguistic quality
    # instead of cutting wherever the char/duration budget runs out.
    break_score_before: int = 0


@dataclass
class ChunkingPolicy:
    """Configurable chunking parameters per language."""

    max_tokens: int = 10
    max_chars: int = 42  # Match max_chars_per_line so chunks fit on one line
    min_chunk_tokens: int = 2
    # Dependency relations that must stay with their head (NEVER split - hard constraint)
    protect_deps: Set[str] = field(
        default_factory=lambda: {
            "aux",
            "cop",
            "compound:prt",
            "compound",
            "fixed",
            "case",
            "det",
            "neg",
            "prt",
        }
    )
    # POS sequences that form atomic units
    protect_pos_sequences: List[List[str]] = field(
        default_factory=lambda: [
            ["AUX", "PART", "VERB"],
            ["AUX", "VERB"],
            ["ADP", "DET", "NOUN"],
            ["ADP", "NOUN"],
        ]
    )
    # Dependencies where breaks are preferred (negative = bonus for breaking)
    prefer_break_deps: Set[str] = field(
        default_factory=lambda: {"punct", "cc", "mark", "parataxis"}
    )
    # Soft penalties for breaking certain relations (discouraged but allowed)
    # These are SOFT constraints - breaks are allowed when necessary for readability
    # Values are VERY LOW (5-10) to prioritize synchronization over grammar
    soft_penalty_deps: Dict[str, int] = field(
        default_factory=lambda: {
            "amod": 5,  # Adjective + noun (mild preference to keep together)
            "nsubj": 5,  # Pronoun/subject + verb (okay to break if needed)
            "nummod": 10,  # Number + unit (slightly higher - numbers look bad split)
            "advmod": 5,  # Adverb + verb (e.g., "not possible")
        }
    )
    # Break point bonuses (negative = good place to break)
    break_bonuses: Dict[str, int] = field(
        default_factory=lambda: {
            "punct": -30,  # After punctuation
            "cc": -20,  # Before coordinating conjunction
            "mark": -25,  # Before subordinator (that, which, because)
            "parataxis": -20,  # Parenthetical/aside
            "advcl": -15,  # Before adverbial clause
            "ccomp": -15,  # Before clausal complement
        }
    )
    # Orphan word penalty (single short word alone on line)
    orphan_penalty: int = 100
    # Penalty for ending line with article/preposition
    bad_line_ending_penalty: int = 40


@dataclass
class PackedCue:
    """A subtitle cue packed from one or more chunks."""

    start: float
    end: float
    chunks: List[GrammarChunk]
    lines: List[str]
    cps: float  # Characters per second
    cpl_max: int  # Max chars per line
    violations: List[str]  # QC violations detected

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line) for line in self.lines)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "lines": self.lines,
            "cps": round(self.cps, 2),
        }


# Language-specific chunking policies
# Note: soft_penalty_deps values are VERY LOW (5-10) to prioritize synchronization
# over grammar. Hard constraints are in protect_deps (these NEVER break).
# Synchronization with source timing takes priority over soft grammar preferences.
CHUNKING_POLICIES: Dict[str, ChunkingPolicy] = {
    "de": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,  # Fit on one line (42 chars max)
        protect_deps={
            "aux",
            "cop",
            "compound:prt",
            "fixed",
            "case",
            "det",
            "neg",
        },
        protect_pos_sequences=[["AUX", "PART", "VERB"], ["ADP", "DET", "NOUN"]],
        soft_penalty_deps={
            "amod": 5,  # Reduced from 20 - mild preference for adj+noun
            "nsubj": 5,  # Reduced from 15 - mild preference for subj+verb
            "nummod": 10,  # Reduced from 20 - keep number+unit
            "advmod": 5,  # Reduced from 10 - adverb+verb
        },
    ),
    "en": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,  # Fit on one line
        protect_deps={"aux", "cop", "compound", "fixed", "case", "det", "neg"},
        soft_penalty_deps={
            "amod": 5,
            "nsubj": 5,
            "nummod": 10,
            "advmod": 5,
        },
    ),
    "fr": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,
        protect_deps={"aux", "cop", "fixed", "case", "det", "neg"},
        soft_penalty_deps={
            "amod": 5,
            "nsubj": 5,
            "nummod": 10,
            "advmod": 5,
        },
    ),
    "es": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,
        protect_deps={"aux", "cop", "fixed", "case", "det", "neg"},
        soft_penalty_deps={
            "amod": 5,
            "nsubj": 5,
            "nummod": 10,
            "advmod": 5,
        },
    ),
    "pt": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,
        protect_deps={"aux", "cop", "fixed", "case", "det", "neg"},
        soft_penalty_deps={
            "amod": 5,
            "nsubj": 5,
            "nummod": 10,
            "advmod": 5,
        },
    ),
    "it": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,
        protect_deps={"aux", "cop", "fixed", "case", "det", "neg"},
        soft_penalty_deps={
            "amod": 5,
            "nsubj": 5,
            "nummod": 10,
            "advmod": 5,
        },
    ),
    "nl": ChunkingPolicy(
        max_tokens=10,
        max_chars=42,
        protect_deps={"aux", "cop", "compound:prt", "fixed", "case", "det", "neg"},
        soft_penalty_deps={
            "amod": 5,
            "nsubj": 5,
            "nummod": 10,
            "advmod": 5,
        },
    ),
    "default": ChunkingPolicy(),  # Fallback for other languages
}


# Language-specific abbreviations that should not be treated as sentence endings
ABBREVIATIONS: Dict[str, Set[str]] = {
    "en": {
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Jr.",
        "Sr.",
        "Inc.",
        "Ltd.",
        "Corp.",
        "Co.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "cf.",
        "St.",
        "Ave.",
        "Blvd.",
        "Rd.",
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Oct.",
        "Nov.",
        "Dec.",
        "Gen.",
        "Col.",
        "Lt.",
        "Sgt.",
        "Rev.",
        "Vol.",
        "No.",
        "Fig.",
        "approx.",
        "dept.",
    },
    "de": {
        "Hr.",
        "Fr.",
        "Dr.",
        "Prof.",
        "Nr.",
        "z.B.",
        "d.h.",
        "u.a.",
        "usw.",
        "bzw.",
        "ca.",
        "ggf.",
        "inkl.",
        "max.",
        "min.",
        "evtl.",
        "Str.",
        "Bd.",
        "Jh.",
        "Mio.",
        "Mrd.",
    },
    "fr": {
        "M.",
        "Mme.",
        "Mlle.",
        "Dr.",
        "Prof.",
        "Jr.",
        "Sr.",
        "St.",
        "etc.",
        "vol.",
        "fig.",
        "env.",
        "av.",
    },
    "es": {
        "Sr.",
        "Sra.",
        "Srta.",
        "Dr.",
        "Dra.",
        "Prof.",
        "etc.",
        "Ud.",
        "Uds.",
        "pág.",
        "núm.",
        "vol.",
        "fig.",
        "aprox.",
    },
    "pt": {
        "Sr.",
        "Sra.",
        "Dr.",
        "Dra.",
        "Prof.",
        "etc.",
        "pág.",
        "núm.",
        "vol.",
        "fig.",
        "aprox.",
    },
}

# Maximum cues to include in a single sentence group
MAX_CUES_PER_SENTENCE = 8

# Function words that must not dangle at the end of a cue or rendered line
# (Netflix TTSG / BBC line-treatment rules: don't separate an article,
# preposition, possessive, auxiliary, conjunction or subject pronoun from what
# follows it). Lowercase, punctuation-stripped comparison. Keyed by primary
# language subtag; unknown languages fall back to "en" — these are Latin-script
# closed-class words, so a false hit on another language is harmless while a
# missing set would disable the guard entirely.
LINE_TAIL_FUNCTION_WORDS: Dict[str, frozenset] = {
    "en": frozenset("""
        a an the my your his her its our their this that these those
        of to in on at for with by from as into onto over under about
        and or but nor so than because if when while although
        is are was were be been being am do does did have has had
        will would can could shall should may might must not
        i we you he she it they who which whose whom there
        i'm we're you're it's that's don't doesn't isn't aren't won't
        """.split()),
    "de": frozenset("""
        der die das dem den des ein eine einem einen einer eines
        mein dein sein ihr unser euer kein keine keinem keinen keiner
        und oder aber sondern denn dass daß weil wenn als ob obwohl
        zu in an auf für mit von bei aus nach über unter vor hinter
        durch gegen ohne um zwischen
        ist sind war waren sein bin bist seid
        hat haben hatte hatten habe hast habt
        wird werden wurde wurden würde würden
        kann können konnte muss müssen musste soll sollen sollte
        will wollen wollte darf dürfen mag mögen möchte
        nicht sich es man ich du er wir sie ihr
        """.split()),
    "fr": frozenset("""
        le la les un une des du de d' à au aux
        mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs
        ce cet cette ces
        et ou mais donc or ni car que qui dont où si quand comme
        dans sur sous avec pour par sans chez vers entre
        est sont était étaient être suis es êtes sommes
        a ont avait avaient ai as avez avons
        sera seront serait peut peuvent pouvait doit doivent devait
        va vont allait ne pas plus très
        je tu il elle on nous vous ils elles se y en
        """.split()),
    "es": frozenset("""
        el la los las un una unos unas lo
        mi mis tu tus su sus nuestro nuestra nuestros nuestras
        este esta estos estas ese esa esos esas
        y e o u pero sino que quien cuando como si porque aunque
        de a en con por para sin sobre entre hacia desde hasta
        es son era eran ser soy eres somos sois está están estaba
        ha han había he has hemos habéis
        será serán sería puede pueden podía debe deben debía
        va van iba no muy
        yo tú él ella ellos ellas nosotros vosotros ustedes usted se me te le nos os les
        """.split()),
    "pt": frozenset("""
        o a os as um uma uns umas
        meu minha meus minhas teu tua seu sua seus suas nosso nossa nossos nossas
        este esta estes estas esse essa esses essas isto isso
        e ou mas porém que quem quando como se porque embora
        de do da dos das em no na nos nas com por para sem sobre entre até
        é são era eram ser sou és somos sois está estão estava
        tem têm tinha havia há
        será serão seria pode podem podia deve devem devia
        vai vão ia não muito
        eu tu ele ela eles elas nós vós você vocês me te lhe lhes
        """.split()),
}


def _function_words_for(lang: Optional[str]) -> frozenset:
    """Function-word set for a language code ('de', 'de-DE', 'de_DE' → 'de')."""
    primary = (lang or "en").replace("_", "-").split("-")[0].lower()
    return LINE_TAIL_FUNCTION_WORDS.get(primary, LINE_TAIL_FUNCTION_WORDS["en"])


_TAIL_STRIP_CHARS = "\"'”’»)]}"


def _ends_on_function_word(text: str, function_words: frozenset) -> bool:
    """True when ``text`` ends on a bare function word — i.e. a linguistically
    bad place to break. Trailing punctuation (other than closing quotes/
    brackets) means the speaker completed a unit there, so it is never a bad
    tail — EXCEPT a dash glued to a function word ("but if the—"), which is a
    speech-repair marker, not a boundary."""
    tail = text.strip().rstrip(_TAIL_STRIP_CHARS).rstrip("—–-")
    if not tail:
        return False
    if not (tail[-1].isalnum() or tail[-1] == "'"):
        return False  # ends with real punctuation — legitimate break
    last = tail.split()[-1].strip(_TAIL_STRIP_CHARS).lower()
    return last in function_words


@dataclass
class PackAtom:
    """An indivisible unit for cue packing: a grammar chunk (grammar path) or a
    single word (basic path). The DP packer partitions a sentence's atoms into
    cues; a cue boundary can only fall between atoms."""

    text: str
    start: float
    end: float
    chars: int
    n_words: int
    # Linguistic score for breaking BEFORE this atom (lower = better break).
    break_score_before: int = 0
    speaker: Optional[str] = None


class SubtitleOptimizer:
    """
    Optimizes word-level transcription data into readable subtitles.

    Default settings follow broadcast subtitle standards:
    - Max 42 characters per line
    - Max 2 lines per cue
    - Target CPS: 15-21 (comfortable reading speed)
    - Minimum cue duration: 1.0 second
    - Maximum cue duration: 7.0 seconds
    """

    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines_per_cue: int = 2,
        target_cps: float = 18.0,
        max_cps: float = 21.0,
        min_cue_duration: float = 1.0,
        max_cue_duration: float = 7.0,
        pause_threshold: float = 0.3,  # 300ms pause triggers new cue
        long_pause_threshold: float = 0.7,  # 700ms forces new cue
        merge_short_cues: bool = True,
        merge_short_min_chars: int = 20,
    ):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_cue = max_lines_per_cue
        self.target_cps = target_cps
        self.max_cps = max_cps
        self.min_cue_duration = min_cue_duration
        self.max_cue_duration = max_cue_duration
        self.pause_threshold = pause_threshold
        self.long_pause_threshold = long_pause_threshold
        # Post-pass: fold orphan/short cues (e.g. a stranded "things." or a
        # lone "So") back into an adjacent same-speaker cue when the merge
        # still respects the line / duration / CPS limits. Providers with
        # precise word timings + aggressive punctuation (AssemblyAI) otherwise
        # leave short tails that read badly.
        self.merge_short_cues = merge_short_cues
        self.merge_short_min_chars = merge_short_min_chars

        # Punctuation that suggests good break points
        self.strong_break_punct = {".", "!", "?", ":", ";"}
        self.soft_break_punct = {",", "–", "—", "-"}

        # Language of the words currently being optimized. Set by
        # optimize_grammar_aware(); drives the function-word guards in cue
        # packing and line balancing. "en" is a safe default for the guards
        # (closed-class Latin-script words).
        self.active_lang = "en"

    # ------------------------------------------------------------------
    # DP cue packing (cost model)
    #
    # Cue boundaries are chosen by minimizing a global cost over each
    # sentence's atoms (grammar chunks, or words on the fallback path)
    # instead of greedily cutting when a char/duration budget overflows.
    # Weights are class attributes so tests can pin/probe them.
    # ------------------------------------------------------------------
    PACK_COST_PER_CUE = 60.0  # prefer fewer cues, all else equal
    PACK_COST_CPS_HARD = 900.0  # base penalty for exceeding max_cps
    PACK_COST_CPS_HARD_SLOPE = 60.0  # per cps above max_cps
    PACK_COST_CPS_SOFT_QUAD = 4.0  # quadratic above target_cps
    PACK_COST_UNDERFULL = 170.0  # scale for non-final underfull cues (head-heavy)
    PACK_COST_ORPHAN_FINAL = 100.0  # short sentence-final cue (tail may be short)
    PACK_COST_ORPHAN = 340.0  # short non-final cue ("So a part of my")
    PACK_COST_SHORT_DURATION = 90.0  # cue under min_cue_duration
    PACK_SYNTAX_WEIGHT = 3.0  # multiplier on chunker break scores
    PACK_SYNTAX_CLAMP = (-40.0, 80.0)  # clamp raw break scores before weighting
    # Strong enough that a ~1s hesitation overrides a mildly bad syntax tail
    # AND the head-fullness preference — a subtitle should not bridge a real
    # silence mid-sentence when a boundary can sit on it.
    PACK_PAUSE_BONUS_PER_SEC = 140.0  # boundary bonus per second of silence
    PACK_PAUSE_BONUS_CAP_SEC = 1.5  # silence credited at most this long
    PACK_PAUSE_MIN_SEC = 0.25  # gaps below this are alignment noise
    PACK_ORPHAN_MIN_CHARS = 20  # a cue smaller than this is an orphan
    PACK_ORPHAN_MAX_WORDS = 3  # ... or with this few words
    PACK_FN_TAIL_PENALTY = 120.0  # non-final cue ending on a dangling function word
    # ...waived when a real hesitation sits at the boundary: the speaker
    # dangled there, so the subtitle mirroring it is correct ("…research is |
    # [pause] early childhood education").
    PACK_FN_TAIL_PAUSE_WAIVER_SEC = 0.6
    LINE_TAIL_PENALTY = 26  # line-break score for ending a line on a function word

    def _atom_group_cost(
        self,
        atoms: List[PackAtom],
        i: int,
        j: int,
        is_last: bool,
    ) -> Optional[float]:
        """Cost of putting atoms[i..j] (inclusive) into one cue, or None when
        infeasible. Boundary cost for the break AFTER j is included unless the
        group is sentence-final."""
        max_chars = self.max_chars_per_line * self.max_lines_per_cue

        chars = sum(a.chars for a in atoms[i : j + 1]) + (j - i)
        duration = atoms[j].end - atoms[i].start
        n_words = sum(a.n_words for a in atoms[i : j + 1])

        if j > i:
            if chars > max_chars:
                return None
            if duration > self.max_cue_duration:
                return None
            # A diarization speaker change is a hard cue boundary.
            for k in range(i, j):
                sa, sb = atoms[k].speaker, atoms[k + 1].speaker
                if sa is not None and sb is not None and sa != sb:
                    return None

        cost = self.PACK_COST_PER_CUE

        cps = chars / max(duration, 0.3)
        if cps > self.max_cps:
            cost += self.PACK_COST_CPS_HARD + (cps - self.max_cps) * (
                self.PACK_COST_CPS_HARD_SLOPE
            )
        elif cps > self.target_cps:
            cost += (cps - self.target_cps) ** 2 * self.PACK_COST_CPS_SOFT_QUAD

        if duration < self.min_cue_duration:
            cost += self.PACK_COST_SHORT_DURATION

        if not is_last:
            # Underfull non-final cues push words rightward; penalizing them
            # yields the "head gets more words, tail may be short" shape.
            fill_gap = (max_chars - chars) / max_chars
            cost += fill_gap * fill_gap * self.PACK_COST_UNDERFULL

        if chars < self.PACK_ORPHAN_MIN_CHARS or n_words <= self.PACK_ORPHAN_MAX_WORDS:
            cost += self.PACK_COST_ORPHAN_FINAL if is_last else self.PACK_COST_ORPHAN

        if not is_last:
            nxt = atoms[j + 1]
            lo, hi = self.PACK_SYNTAX_CLAMP
            syn = min(max(float(nxt.break_score_before), lo), hi)
            cost += syn * self.PACK_SYNTAX_WEIGHT
            gap = nxt.start - atoms[j].end
            if gap >= self.PACK_PAUSE_MIN_SEC:
                cost -= (
                    min(gap, self.PACK_PAUSE_BONUS_CAP_SEC)
                    * self.PACK_PAUSE_BONUS_PER_SEC
                )
            # Don't let a cue dangle on a function word ("…the topic of |")
            # over a minor gap — dependency scores don't catch every such
            # tail (e.g. ADP before a gerund complement). A real hesitation
            # waives the penalty: the speaker dangled there, the subtitle
            # mirroring it is correct.
            if gap < self.PACK_FN_TAIL_PAUSE_WAIVER_SEC and _ends_on_function_word(
                atoms[j].text, _function_words_for(self.active_lang)
            ):
                cost += self.PACK_FN_TAIL_PENALTY

        return cost

    def _dp_pack_atoms(self, atoms: List[PackAtom]) -> List[Tuple[int, int]]:
        """Partition atoms into cue groups minimizing total cost.

        Returns a list of (i, j) inclusive index ranges. Single atoms are
        always feasible, so a solution always exists.
        """
        n = len(atoms)
        if n == 0:
            return []

        INF = float("inf")
        best = [INF] * (n + 1)  # best[k] = min cost covering atoms[0..k-1]
        back = [0] * (n + 1)
        best[0] = 0.0

        for k in range(1, n + 1):
            j = k - 1
            is_last = k == n
            for i in range(j, -1, -1):
                if best[i] == INF:
                    continue
                cost = self._atom_group_cost(atoms, i, j, is_last)
                if cost is None:
                    # Larger groups only get longer/bigger — stop widening.
                    break
                total = best[i] + cost
                if total < best[k]:
                    best[k] = total
                    back[k] = i

        if best[n] == INF:  # unreachable — singletons are always feasible
            return [(i, i) for i in range(n)]

        groups: List[Tuple[int, int]] = []
        k = n
        while k > 0:
            i = back[k]
            groups.append((i, k - 1))
            k = i
        groups.reverse()
        return groups

    def _dominant_speaker(self, cue: "SubtitleCue"):
        sp = [w.speaker for w in cue.words if w.speaker is not None]
        return max(set(sp), key=sp.count) if sp else None

    def _pack_texts_into_lines(self, text: str):
        """Greedily wrap a text string into ≤ max_lines_per_cue lines of
        ≤ max_chars_per_line. Returns the line list, or None if it can't fit
        (so the caller rejects the merge). Text-based (not Word-based) so it
        works for grammar-chunked cues whose ``words`` list may be empty."""
        lines: List[str] = []
        cur = ""
        for tok in text.split():
            cand = tok if not cur else cur + " " + tok
            if len(cand) <= self.max_chars_per_line:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = tok
        if cur:
            lines.append(cur)
        if len(lines) > self.max_lines_per_cue:
            return None
        return lines

    def _is_short_cue(self, cue: "SubtitleCue") -> bool:
        # char_count comes from ``lines`` — reliable even when ``words`` is
        # empty (grammar-chunked cues carry their text in ``lines``).
        # A cue is "short" by character count OR by word count — "So a part
        # of my" is 15 chars but 5 words; "extraordinarily incomprehensible"
        # is 2 words but plenty of reading material. Both axes must agree
        # that the cue carries real content for it to stand alone.
        if cue.char_count <= 0:
            return False
        if cue.char_count < self.merge_short_min_chars:
            return True
        n_words = len(cue.text.replace("\n", " ").split())
        return n_words <= 2 and cue.char_count < 2 * self.merge_short_min_chars

    def _try_merge_cues(self, a: "SubtitleCue", b: "SubtitleCue"):
        """Merge b into a if safe (compatible speaker, fits lines/duration/CPS).
        Text-based so grammar-chunked cues (possibly empty ``words``) merge
        correctly. Returns the merged cue or None."""
        # Speaker guard: only block when BOTH cues have a known — and
        # different — dominant speaker. Unknown (None) never blocks (covers
        # non-diarized transcripts, where every speaker is None).
        sa, sb = self._dominant_speaker(a), self._dominant_speaker(b)
        if sa is not None and sb is not None and sa != sb:
            return None
        a_text = a.text.replace("\n", " ").strip()
        b_text = b.text.replace("\n", " ").strip()
        # Don't merge across a sentence/clause end — if `a` already closes with
        # strong punctuation, `b` begins a new thought and stays its own cue.
        # This folds an orphan tail ("things.") BACK into its sentence while a
        # following "So …" starts fresh.
        if a_text and a_text[-1] in self.strong_break_punct:
            return None
        combined = (a_text + " " + b_text).strip()
        lines = self._pack_texts_into_lines(combined)
        if lines is None:
            return None
        start = min(a.start, b.start)
        end = max(a.end, b.end)
        duration = max(0.001, end - start)
        if duration > self.max_cue_duration:
            return None
        char_count = sum(len(ln) for ln in lines)
        if char_count / duration > self.max_cps:
            return None
        return SubtitleCue(start=start, end=end, lines=lines, words=a.words + b.words)

    def _merge_short_cues(self, cues: List["SubtitleCue"]) -> List["SubtitleCue"]:
        """Fold stranded short cues into an adjacent cue where it reads better
        and still respects all constraints. Conservative: only merges when a
        cue (or its neighbor) is short and the merged result is within limits."""
        if not self.merge_short_cues or len(cues) < 2:
            return cues
        out: List[SubtitleCue] = []
        for cue in cues:
            if out and (self._is_short_cue(cue) or self._is_short_cue(out[-1])):
                merged = self._try_merge_cues(out[-1], cue)
                if merged is not None:
                    out[-1] = merged
                    continue
            out.append(cue)
        return out

    def optimize(self, words: List[Word]) -> List[SubtitleCue]:
        """
        Convert a list of words with timestamps into optimized subtitle cues.

        Algorithm:
        1. Group words into potential cues based on pauses and punctuation
        2. Enforce character limits by splitting long groups
        3. Balance lines within each cue
        4. Ensure timing constraints (min/max duration, CPS)
        """
        if not words:
            return []

        # Step 1: Initial segmentation based on pauses and punctuation
        raw_cues = self._segment_by_pauses_and_punctuation(words)

        # Step 2+3: Partition each segment into cues via the DP packer
        # (pause + punctuation + function-word aware), then balance lines.
        final_cues = []
        for cue_words in raw_cues:
            final_cues.extend(self._pack_words_to_cues(cue_words))

        # Step 4: Apply timing adjustments
        final_cues = self._adjust_timing(final_cues)

        # Step 5: Fold stranded short cues into a neighbour where safe.
        final_cues = self._merge_short_cues(final_cues)
        final_cues = self._enforce_line_width(final_cues)

        return final_cues

    def _segment_by_pauses_and_punctuation(self, words: List[Word]) -> List[List[Word]]:
        """
        Group words into segments based on:
        - Long pauses (force break)
        - Short pauses + punctuation (prefer break)
        - Maximum duration limits
        """
        if not words:
            return []

        segments = []
        current_segment = [words[0]]

        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]

            # Calculate pause between words
            pause = curr_word.start - prev_word.end

            # Check if we should start a new segment
            should_break = False

            # A long pause forces a break — but a hesitation pause right after
            # a dangling function word ("So a part of my …") must not cut
            # mid-phrase. Those mid-phrase pauses stay inside the segment and
            # the DP packer scores them as boundary candidates instead. Only
            # a pause with completed phrasing (any punctuation) or true
            # silence (2x threshold) cuts here.
            if pause >= self.long_pause_threshold:
                prev_text = prev_word.word.strip()
                phrase_done = bool(prev_text) and not (
                    prev_text[-1].isalnum() or prev_text[-1] == "'"
                )
                if phrase_done or pause >= 2 * self.long_pause_threshold:
                    should_break = True

            # Short pause + strong punctuation = break
            elif pause >= self.pause_threshold:
                if self._ends_with_strong_punct(prev_word.word):
                    should_break = True

            # Strong punctuation alone with reasonable pause
            elif pause >= 0.1 and self._ends_with_strong_punct(prev_word.word):
                # Check if segment is getting long
                segment_text = " ".join(w.word for w in current_segment)
                if len(segment_text) > self.max_chars_per_line:
                    should_break = True

            # Max duration exceeded
            if not should_break and current_segment:
                segment_duration = curr_word.start - current_segment[0].start
                if segment_duration >= self.max_cue_duration:
                    should_break = True

            if should_break:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [curr_word]
            else:
                current_segment.append(curr_word)

        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)

        return segments

    def _word_break_score(self, prev_word: Word, function_words: frozenset) -> int:
        """Heuristic linguistic score for breaking AFTER prev_word on the
        no-parser path (lower = better), mirroring the grammar chunker's
        scale: punctuation is a good boundary, a dangling function word a
        bad one."""
        text = prev_word.word.strip()
        if self._ends_with_strong_punct(text):
            return -35
        # A dash/comma glued to a dangling function word is a speech repair
        # ("but if the—", "of the,"), not a boundary — check the word core
        # before crediting the punctuation.
        core_is_function = _ends_on_function_word(
            text.rstrip("".join(self.soft_break_punct)), function_words
        )
        if self._ends_with_soft_punct(text) and not core_is_function:
            return -25
        if core_is_function:
            # Kept below the ~1s-pause bonus: a genuine hesitation after a
            # dangling function word is still a better break than bridging
            # the silence (the orphan cost separately blocks tiny heads).
            return 25
        return 0

    def _pack_words_to_cues(self, words: List[Word]) -> List[SubtitleCue]:
        """Partition a word group into cues with the DP packer (fallback path
        when no dependency parse is available). Break candidates are scored by
        punctuation / function-word heuristics plus the shared pause and
        readability cost model."""
        if not words:
            return []

        function_words = _function_words_for(self.active_lang)
        atoms: List[PackAtom] = []
        for i, w in enumerate(words):
            text = w.word.strip()
            atoms.append(
                PackAtom(
                    text=text,
                    start=w.start,
                    end=w.end,
                    chars=len(text),
                    n_words=1,
                    break_score_before=(
                        self._word_break_score(words[i - 1], function_words)
                        if i > 0
                        else 0
                    ),
                    speaker=w.speaker,
                )
            )

        cues = []
        for i, j in self._dp_pack_atoms(atoms):
            cue = self._build_cue(words[i : j + 1])
            if cue:
                cues.append(cue)
        return cues

    def _build_cue(self, words: List[Word]) -> Optional[SubtitleCue]:
        """
        Build a subtitle cue from a list of words.
        Balances text across 1-2 lines.
        """
        if not words:
            return None

        full_text = " ".join(w.word.strip() for w in words)

        # Determine timing using span approach (min start, max end)
        start = min(w.start for w in words)
        end = max(w.end for w in words)

        # If text fits on one line, use one line
        if len(full_text) <= self.max_chars_per_line:
            lines = [full_text]
        else:
            # Split into two lines
            lines = self._balance_two_lines(words)

        return SubtitleCue(
            start=start,
            end=end,
            lines=lines,
            words=words,
        )

    def _enforce_line_width(self, cues: List[SubtitleCue]) -> List[SubtitleCue]:
        """Final guard: no emitted line may exceed max_chars_per_line.

        Cues reach this point from several construction paths (pause
        segmentation, short-cue merging, chunk finalisation), and not all of them
        route through the two-line balancer. A cue that arrives as a single
        over-budget line would otherwise ship as-is, which breaks the one
        guarantee callers rely on.

        Re-splits any over-budget line on word boundaries. A single word longer
        than the budget is left alone -- there is nowhere to break it, and
        silently truncating would lose content.
        """
        for cue in cues:
            if all(len(line) <= self.max_chars_per_line for line in cue.lines):
                continue

            tokens = " ".join(cue.lines).split()
            rebuilt: List[str] = []
            current = ""
            for token in tokens:
                candidate = f"{current} {token}".strip()
                if current and len(candidate) > self.max_chars_per_line:
                    rebuilt.append(current)
                    current = token
                else:
                    current = candidate
            if current:
                rebuilt.append(current)

            # Respect the two-line cap. If re-splitting produced more lines than
            # a cue may carry, keep the balancer's original output rather than
            # emit a cue that breaks a different invariant -- an over-long line
            # is a readability problem, a three-line cue is a rendering one.
            if rebuilt and len(rebuilt) <= self.max_lines_per_cue:
                cue.lines = rebuilt
        return cues

    def _balance_two_lines(self, words: List[Word]) -> List[str]:
        """
        Split words into two balanced lines.
        Prefers splitting at punctuation, then at roughly equal character counts.
        """
        if not words:
            return []

        full_text = " ".join(w.word.strip() for w in words)

        if len(full_text) <= self.max_chars_per_line:
            return [full_text]

        # Find the best split point. Linguistic quality outranks geometry: a
        # line must not end on a dangling function word ("So a part of my /
        # research is …" reads worse than a lopsided split). Balance is the
        # tiebreaker.
        function_words = _function_words_for(self.active_lang)
        best_split = len(words) // 2
        best_diff = float("inf")

        for i in range(1, len(words)):
            line1 = " ".join(w.word.strip() for w in words[:i])
            line2 = " ".join(w.word.strip() for w in words[i:])

            # Skip if either line is too long
            if (
                len(line1) > self.max_chars_per_line
                or len(line2) > self.max_chars_per_line
            ):
                continue

            diff = abs(len(line1) - len(line2))

            # Bonus for splitting after punctuation; penalty for a dangling
            # function word at the line break.
            prev_word = words[i - 1].word.strip()
            if self._ends_with_strong_punct(prev_word):
                diff -= 10
            elif self._ends_with_soft_punct(prev_word):
                diff -= 5
            elif _ends_on_function_word(prev_word, function_words):
                diff += self.LINE_TAIL_PENALTY

            if diff < best_diff:
                best_diff = diff
                best_split = i

        line1 = " ".join(w.word.strip() for w in words[:best_split])
        line2 = " ".join(w.word.strip() for w in words[best_split:])

        # Don't truncate - let human editor handle overflow
        lines = []
        if line1:
            lines.append(line1)
        if line2:
            lines.append(line2)

        return lines if lines else [full_text]

    def _adjust_timing(self, cues: List[SubtitleCue]) -> List[SubtitleCue]:
        """
        Adjust cue timing for readability:
        - Enforce minimum duration
        - Extend over-CPS cues into the following silence (standard
          remediation: give the viewer the missing reading time when the
          audio timeline has room)
        - Add small gaps between cues
        - Prevent overlaps
        """
        if not cues:
            return cues

        min_gap = 0.05  # 50ms minimum gap between cues

        for i, cue in enumerate(cues):
            # Enforce minimum duration
            if cue.duration < self.min_cue_duration:
                cue.end = cue.start + self.min_cue_duration

            # Reading-speed remediation: when the cue exceeds max_cps, extend
            # its end so cps drops to max (bounded below by max_cue_duration
            # and, for non-final cues, by the next cue's start).
            if cue.cps > self.max_cps:
                needed_end = cue.start + cue.char_count / self.max_cps
                cue.end = min(needed_end, cue.start + self.max_cue_duration)

            # Prevent overlap with next cue
            if i < len(cues) - 1:
                next_cue = cues[i + 1]
                if cue.end >= next_cue.start - min_gap:
                    cue.end = max(cue.start + 0.2, next_cue.start - min_gap)

        return cues

    def _ends_with_strong_punct(self, text: str) -> bool:
        """Check if text ends with strong punctuation."""
        text = text.strip()
        return bool(text) and text[-1] in self.strong_break_punct

    def _ends_with_soft_punct(self, text: str) -> bool:
        """Check if text ends with soft punctuation."""
        text = text.strip()
        return bool(text) and text[-1] in self.soft_break_punct

    def optimize_grammar_aware(
        self, words: List[Word], lang: str = "en"
    ) -> List[SubtitleCue]:
        """
        Convert words into subtitle cues using GRAMMAR-AWARE chunking.

        Unlike the basic optimize() which uses pauses and punctuation,
        this method uses spaCy dependency parsing to ensure grammatical
        units are kept together (e.g., "der große" won't be split).

        Algorithm:
        1. Group words into sentences using punctuation
        2. Parse each sentence with spaCy to get POS/dependency info
        3. Align spaCy tokens to ASR words (handling tokenization differences)
        4. Use grammar chunker to find safe break points
        5. Map chunks back to word-level timestamps
        6. Pack into cues respecting CPL/CPS constraints

        Args:
            words: List of Word objects with ASR timing
            lang: Language code for spaCy model (de, en, fr, es, pt, it, nl)

        Returns:
            List of optimized SubtitleCue objects
        """
        if not words:
            return []

        import logging

        logger = logging.getLogger(__name__)

        # Remember the language for the function-word guards used by cue
        # packing and line balancing (also on the basic fallback path below).
        self.active_lang = lang or "en"

        # Check if parser is available for this language
        provider = get_grammar_provider()
        if not provider.can_parse:
            logger.info(
                "No grammar provider registered; using the dependency-free "
                "punctuation/pause packer. See subtitle_optimizer.grammar."
            )
            return self.optimize(words)

        parse_sentence = provider.parse_sentence
        is_parser_available = provider.is_parser_available or (lambda _l: True)

        if not is_parser_available(lang):
            logger.info(
                "Grammar parsing not available for %s, using basic optimize", lang
            )
            return self.optimize(words)

        chunk_sentence = provider.chunk_sentence

        # Step 1: Group words into sentences
        sentences = self._group_words_into_sentences(words, lang=lang)
        logger.info(
            "Grouped %d words into %d sentences for grammar-aware chunking",
            len(words),
            len(sentences),
        )

        all_cues: List[SubtitleCue] = []

        for sent_words in sentences:
            if not sent_words:
                continue

            # Build sentence text
            sent_text = " ".join(w.word.strip() for w in sent_words)

            # Step 2: Parse with spaCy
            parse_result = parse_sentence(sent_text, lang)
            if not parse_result:
                # Fallback to basic chunking for this sentence
                cues = self._chunk_sentence_basic(sent_words)
                all_cues.extend(cues)
                continue

            # Step 3: Align spaCy tokens to ASR words
            aligned_tokens, word_to_token_map = self._align_spacy_to_asr(
                parse_result, sent_words
            )

            if not aligned_tokens:
                cues = self._chunk_sentence_basic(sent_words)
                all_cues.extend(cues)
                continue

            # Build ParsedSentence
            parsed = ParsedSentence(
                text=sent_text,
                tokens=aligned_tokens,
                lang=lang,
                parse_model=f"spacy_{lang}",
            )

            # Step 4: Grammar-chunk the sentence. Pass the inter-word silences
            # (keyed by spaCy token index) so the chunker's break scoring can
            # prefer boundaries that sit on real pauses — a hesitation after
            # "is" beats a 50ms gap after "they" even when both are
            # syntactically allowed.
            boundary_pauses: Dict[int, float] = {}
            for tok in aligned_tokens:
                first_words = word_to_token_map.get(tok.idx) or []
                if not first_words:
                    continue
                w_idx = min(first_words)
                if w_idx <= 0 or w_idx >= len(sent_words):
                    continue
                gap = sent_words[w_idx].start - sent_words[w_idx - 1].end
                if gap > 0:
                    boundary_pauses[tok.idx] = gap

            chunks = chunk_sentence(parsed, lang, boundary_pauses=boundary_pauses)

            if not chunks:
                cues = self._chunk_sentence_basic(sent_words)
                all_cues.extend(cues)
                continue

            # Step 5: Map chunks back to whole ASR words and rebuild their text
            # from those words. spaCy re-tokenises sub-word pieces ("white-centric"
            # -> "white", "-", "centric"; "it's" -> "it", "'s") and a chunk (hence
            # cue) boundary can fall between them — corrupting the output as
            # "white - centric" or splitting a sentence mid-word ("...say it" |
            # "'s also..."). The ASR words are the source of truth: each is claimed
            # by exactly the first chunk that references any of its spaCy tokens, so
            # no word is ever split across chunks, and each chunk's text is rebuilt
            # from its intact ASR words rather than joined spaCy fragments.
            assigned_word_ids: set = set()
            for chunk in chunks:
                chunk_words = []
                for token_idx in chunk.token_indices:
                    if token_idx not in word_to_token_map:
                        continue
                    for word_idx in word_to_token_map[token_idx]:
                        if word_idx >= len(sent_words):
                            continue
                        if word_idx in assigned_word_ids:
                            continue  # already owned by an earlier chunk
                        assigned_word_ids.add(word_idx)
                        chunk_words.append((word_idx, sent_words[word_idx]))

                if not chunk_words:
                    # Chunk held only sub-word fragments already owned by a
                    # neighbour (e.g. a stray "'s"). Blank it so it is dropped
                    # before packing and its acoustic span is absorbed elsewhere.
                    chunk.text = ""
                    chunk.char_count = 0
                    chunk.start_time = None
                    chunk.end_time = None
                    continue

                chunk_words.sort(key=lambda pair: pair[0])
                ordered = [w for _, w in chunk_words]
                chunk.text = join_texts_naturally([w.word.strip() for w in ordered])
                chunk.char_count = len(chunk.text)
                chunk.start_time = min(w.start for w in ordered)
                chunk.end_time = max(w.end for w in ordered)

            # Step 6: Pack chunks into cues (drop chunks emptied above)
            packable = [c for c in chunks if c.text]
            cues = self._pack_grammar_chunks_to_cues(packable, sent_words)
            all_cues.extend(cues)

        # Final timing adjustments
        all_cues = self._adjust_timing(all_cues)

        # Fold stranded short cues (orphan tail words, lone connectives) into a
        # neighbouring same-speaker cue where the merge stays within limits.
        all_cues = self._merge_short_cues(all_cues)

        return all_cues

    def _group_words_into_sentences(
        self, words: List[Word], lang: str = "en"
    ) -> List[List[Word]]:
        """
        Group words into sentences based on sentence-ending punctuation.
        """
        if not words:
            return []

        abbreviations = ABBREVIATIONS.get(lang, ABBREVIATIONS.get("en", set()))
        sentences = []
        current = []

        for word in words:
            current.append(word)
            current_text = " ".join(w.word.strip() for w in current).strip()

            # Abbreviation-aware sentence boundary detection so initials like
            # "T." and titles like "St." do not trigger premature sentence splits.
            if _is_sentence_end(current_text, abbreviations):
                sentences.append(current)
                current = []

        # Don't forget remaining words
        if current:
            sentences.append(current)

        return sentences

    def _align_spacy_to_asr(
        self,
        parse_result: List[dict],
        words: List[Word],
    ) -> Tuple[List[AlignedToken], Dict[int, List[int]]]:
        """
        Align spaCy tokens to ASR words.

        spaCy and ASR may tokenize differently (e.g., "don't" vs "do" + "n't").
        This function creates a mapping between them using character offsets.

        Returns:
            Tuple of (aligned_tokens, word_to_token_map)
            word_to_token_map: {token_idx: [word_indices]} for reverse lookup
        """
        if not parse_result or not words:
            return [], {}

        # Build character offset map for ASR words
        # Reconstruct the sentence and track word boundaries
        asr_text = ""
        word_char_ranges = []  # [(start, end), ...] for each word

        for w in words:
            word_text = w.word.strip()
            if asr_text:
                asr_text += " "
            start = len(asr_text)
            asr_text += word_text
            end = len(asr_text)
            word_char_ranges.append((start, end))

        aligned_tokens = []
        word_to_token_map: Dict[int, List[int]] = {}

        for tok in parse_result:
            tok_start = tok["char_start"]
            tok_end = tok["char_end"]

            # Find which ASR words overlap with this token
            aligned_word_indices = []
            for word_idx, (w_start, w_end) in enumerate(word_char_ranges):
                # Check for overlap
                if tok_start < w_end and tok_end > w_start:
                    aligned_word_indices.append(word_idx)

            # Create AlignedToken
            aligned_token = AlignedToken(
                idx=tok["idx"],
                text=tok["text"],
                char_start=tok_start,
                char_end=tok_end,
                aligned_source_indices=aligned_word_indices,  # For originals, source = ASR words
                pos=tok["pos"],
                dep=tok["dep"],
                head_idx=tok["head_idx"],
                is_aligned=bool(aligned_word_indices),
            )
            aligned_tokens.append(aligned_token)

            # Build reverse map
            word_to_token_map[tok["idx"]] = aligned_word_indices

        return aligned_tokens, word_to_token_map

    def _chunk_sentence_basic(self, words: List[Word]) -> List[SubtitleCue]:
        """
        Fallback: chunk a sentence without a dependency parse, via the DP
        packer with punctuation / function-word / pause heuristics.
        """
        return self._pack_words_to_cues(words)

    def _dominant_speaker_in_window(
        self, words: List[Word], start: float, end: float
    ) -> Optional[str]:
        sp = [
            w.speaker
            for w in words
            if w.speaker is not None and start <= w.start <= end
        ]
        return max(set(sp), key=sp.count) if sp else None

    def _pack_grammar_chunks_to_cues(
        self,
        chunks: List[GrammarChunk],
        words: List[Word],
    ) -> List[SubtitleCue]:
        """
        Pack grammar chunks into subtitle cues by globally minimizing a cost
        over candidate boundaries (syntax score, pause alignment, CPS,
        duration, fullness) instead of greedily cutting where the char or
        duration budget happens to overflow.
        """
        if not chunks:
            return []

        atoms: List[PackAtom] = []
        for chunk in chunks:
            if chunk.start_time is None or chunk.end_time is None:
                continue
            atoms.append(
                PackAtom(
                    text=chunk.text,
                    start=chunk.start_time,
                    end=chunk.end_time,
                    chars=chunk.char_count,
                    n_words=len(chunk.text.split()),
                    break_score_before=chunk.break_score_before,
                    speaker=self._dominant_speaker_in_window(
                        words, chunk.start_time, chunk.end_time
                    ),
                )
            )
        if not atoms:
            return []

        packable = [
            c for c in chunks if c.start_time is not None and c.end_time is not None
        ]
        groups = self._dp_pack_atoms(atoms)

        cues = []
        for i, j in groups:
            cue = self._finalize_chunk_cue(packable[i : j + 1], words)
            if cue:
                cues.append(cue)
        return cues

    def pack_translation_chunks(self, chunks: List[GrammarChunk]) -> List["PackedCue"]:
        """
        DP-pack TRANSLATED grammar chunks (projected timestamps) into
        PackedCue objects — the translation-track counterpart of
        `_pack_grammar_chunks_to_cues`, replacing the greedy
        `grammar_chunker.pack_chunks_to_cues`. Set ``active_lang`` to the
        TARGET language first so the function-word tail guard applies to it.

        The chunks arrive already clamped to their sentence-group window and
        monotonic (project_timestamps_whole_sentence), so the DP's atom times
        — and therefore its pause bonuses — reflect the source audio. Timing
        hygiene mirrors the greedy packer's post-passes (min duration,
        monotonic 50ms gaps); QC repair stays with the caller.
        """
        provider = get_grammar_provider()
        _finalize_cue = provider.finalize_cue
        _enforce_min_duration = provider.enforce_min_duration
        _enforce_cue_monotonic_timing = provider.enforce_cue_monotonic_timing
        if not all((_finalize_cue, _enforce_min_duration, _enforce_cue_monotonic_timing)):
            raise RuntimeError(
                "pack_translation_chunks() needs the finalize_cue, "
                "enforce_min_duration and enforce_cue_monotonic_timing hooks. "
                "Register them via subtitle_optimizer.register_grammar_provider()."
            )

        packable = [
            c
            for c in chunks
            if c.text and c.start_time is not None and c.end_time is not None
        ]
        if not packable:
            return []

        atoms = [
            PackAtom(
                text=c.text,
                start=c.start_time,
                end=c.end_time,
                chars=c.char_count,
                n_words=len(c.text.split()),
                break_score_before=c.break_score_before,
            )
            for c in packable
        ]

        cues: List[PackedCue] = []
        for i, j in self._dp_pack_atoms(atoms):
            group = packable[i : j + 1]
            # A cue must not display before its first aligned source word is
            # actually spoken (projection may have stretched the chunk back
            # to the sentence-group start) — same clamp as the greedy packer.
            first = group[0]
            aligned_min = getattr(first, "_aligned_min_start", None)
            if aligned_min is not None and first.start_time < aligned_min:
                first.start_time = aligned_min
                if first.end_time < first.start_time + 0.05:
                    first.end_time = first.start_time + 0.05
            cue = _finalize_cue(
                group,
                first.start_time,
                self.max_chars_per_line,
                self.max_lines_per_cue,
                self.max_cps,
            )
            # The chunk-based line balancer has no function-word guard; when
            # it dangles one ("Ein Teil meiner Forschung ist / …"), re-balance
            # with the guard-aware text balancer (target-language word set).
            fw = _function_words_for(self.active_lang)
            if len(cue.lines) == 2 and _ends_on_function_word(cue.lines[0], fw):
                relined = self._balance_text_to_lines(
                    cue.text.replace("\n", " ").strip()
                )
                if relined and not (
                    len(relined) == 2 and _ends_on_function_word(relined[0], fw)
                ):
                    cue.lines = relined
                    cue.cpl_max = max(len(ln) for ln in relined)
            cues.append(cue)

        _enforce_min_duration(cues, self.min_cue_duration)
        _enforce_cue_monotonic_timing(cues)
        return cues

    def _finalize_chunk_cue(
        self, chunks: List[GrammarChunk], words: Optional[List[Word]] = None
    ) -> Optional[SubtitleCue]:
        """
        Create a SubtitleCue from a list of grammar chunks.
        """
        if not chunks:
            return None

        # Combine chunk texts. Use natural joining (not a blind space) so that a
        # chunk starting with punctuation/contraction/hyphen glues to the prior
        # chunk instead of becoming "... it 's ...".
        text = join_texts_naturally([c.text for c in chunks])

        # Get timing from chunks
        start = min(c.start_time for c in chunks if c.start_time is not None)
        end = max(c.end_time for c in chunks if c.end_time is not None)

        if start is None or end is None:
            return None

        # Balance lines
        if len(text) <= self.max_chars_per_line:
            lines = [text]
        else:
            # Split into two lines at best point
            lines = self._balance_text_to_lines(text)

        # Attach the source words falling in this cue's window so the cue
        # carries its diarization speaker label (cue.to_dict extracts the
        # dominant speaker from words). Without this, grammar-mode cues lose
        # speaker attribution — breaking the transcript editor + RAG
        # speaker-rename. Text/lines are still driven by the chunks above.
        cue_words: List[Word] = []
        if words:
            cue_words = [
                w for w in words if w.start is not None and start <= w.start <= end
            ]

        return SubtitleCue(
            start=start,
            end=end,
            lines=lines,
            words=cue_words,
        )

    def _balance_text_to_lines(self, text: str) -> List[str]:
        """
        Split text into two lines for subtitle display. Linguistic quality
        outranks geometry: never end line 1 on a dangling function word when
        a fitting alternative exists; balance is the tiebreaker. Splits that
        overflow a line are skipped while any fitting split exists.
        """
        if len(text) <= self.max_chars_per_line:
            return [text]

        words = text.split()
        if len(words) <= 1:
            return [text]

        function_words = _function_words_for(self.active_lang)

        best_idx = 0
        best_diff = float("inf")
        # Track the best split ignoring the line-length guard as a fallback
        # (over-long input; the editor handles overflow, we don't truncate).
        fallback_idx = 0
        fallback_diff = float("inf")

        for i in range(1, len(words)):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])

            diff = abs(len(line1) - len(line2))
            last = words[i - 1]
            if last[-1] in ".,;:!?":
                diff -= 10  # Bonus for punctuation
            elif _ends_on_function_word(last, function_words):
                diff += self.LINE_TAIL_PENALTY

            if diff < fallback_diff:
                fallback_diff = diff
                fallback_idx = i

            if (
                len(line1) <= self.max_chars_per_line
                and len(line2) <= self.max_chars_per_line
                and diff < best_diff
            ):
                best_diff = diff
                best_idx = i

        if best_idx == 0:
            best_idx = fallback_idx
        if best_idx == 0:
            return [text]

        line1 = " ".join(words[:best_idx])
        line2 = " ".join(words[best_idx:])

        return [line1, line2] if line1 and line2 else [text]


def words_from_whisper_segments(segments) -> List[Word]:
    """
    Extract Word objects from faster-whisper segment output.

    Args:
        segments: Iterator of segment objects from model.transcribe(word_timestamps=True)

    Returns:
        Flat list of Word objects with timing information
    """
    words = []
    for segment in segments:
        if hasattr(segment, "words") and segment.words:
            for w in segment.words:
                words.append(
                    Word(
                        word=w.word,
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(getattr(w, "probability", 1.0)),
                    )
                )
    return words


def optimize_transcript(words: List[Word], **kwargs) -> List[SubtitleCue]:
    """
    Convenience function to optimize a word list into subtitle cues.

    Args:
        words: List of Word objects with timing
        **kwargs: Optional parameters for SubtitleOptimizer

    Returns:
        List of optimized SubtitleCue objects
    """
    optimizer = SubtitleOptimizer(**kwargs)
    return optimizer.optimize(words)


def cues_to_segments(cues: List[SubtitleCue]) -> List[dict]:
    """
    Convert SubtitleCue objects to the segment dict format used in the database.
    """
    return [cue.to_dict() for cue in cues]


def words_to_raw_segments(words: List[Word]) -> List[dict]:
    """
    Convert Word objects to a raw segment format for storage.
    This preserves the original word-level data — the speaker label when the
    ASR provider returned diarization, and the filler flag for disfluencies
    ("um"/"uh"). Keys are omitted when unset to keep plain segments_json
    identical to before. Fillers are persisted (not dropped) so reprocessing
    keeps the original acoustic timeline.
    """
    out: List[dict] = []
    for w in words:
        d = {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "probability": w.probability,
        }
        if w.speaker is not None:
            d["speaker"] = w.speaker
        if w.is_filler:
            d["filler"] = True
        out.append(d)
    return out


def extract_cues_from_segments_json(segments_json) -> List[dict]:
    """
    Extract subtitle cues from segments_json, handling both old and new formats.

    Old format (list):
        [{"start": 0.0, "end": 2.5, "text": "Hello"}]

    New format (dict with cues and words):
        {"cues": [...], "words": [...], "optimizer_version": "1.0"}

    Returns:
        List of cue dicts with start, end, text (and optionally lines)
    """
    if not segments_json:
        return []

    # New format: dict with "cues" key
    if isinstance(segments_json, dict):
        return segments_json.get("cues", [])

    # Old format: direct list of segments
    if isinstance(segments_json, list):
        return segments_json

    return []


def extract_words_from_segments_json(segments_json) -> List[dict]:
    """
    Extract raw word data from segments_json if available.

    Returns:
        List of word dicts, or empty list if not available (old format)
    """
    if not segments_json:
        return []

    if isinstance(segments_json, dict):
        return segments_json.get("words", [])

    return []


def words_from_raw_segments(raw_words: List[dict]) -> List[Word]:
    """
    Convert raw word dicts back to Word objects for re-optimization.

    Restores the diarization ``speaker`` label too — without it, Reprocess
    silently strips speaker attribution from a diarized transcript — and the
    ``filler`` flag, so reprocess keeps excluding disfluencies from the text
    while their acoustic span survives in the timeline.
    """
    return [
        Word(
            word=w.get("word", ""),
            start=float(w.get("start", 0)),
            end=float(w.get("end", 0)),
            probability=float(w.get("probability", 1.0)),
            speaker=w.get("speaker"),
            is_filler=bool(w.get("filler")),
        )
        for w in raw_words
    ]


def rebalance_lines(
    text: str,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
) -> List[str]:
    """
    Re-balance text into lines respecting character limits.
    Used for translated text where we don't have word-level timestamps.

    Args:
        text: The text to balance into lines
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum number of lines

    Returns:
        List of balanced lines
    """
    if not text:
        return []

    text = text.strip()
    max_total_chars = max_chars_per_line * max_lines

    # If text fits on one line, return as-is
    if len(text) <= max_chars_per_line:
        return [text]

    # Split into words
    words = text.split()
    if not words:
        return [text] if text else []

    # If we need two lines, find the best split point
    # Try to find a split that keeps both lines under the limit
    best_split = len(words) // 2
    best_diff = float("inf")
    found_valid_split = False

    # Punctuation that makes good break points
    good_breaks = {",", ".", "!", "?", ":", ";", "–", "—", "-"}

    for i in range(1, len(words)):
        line1 = " ".join(words[:i])
        line2 = " ".join(words[i:])

        # Prefer splits where both lines fit, but don't require it
        both_fit = len(line1) <= max_chars_per_line and len(line2) <= max_chars_per_line

        diff = abs(len(line1) - len(line2))

        # Bonus for splitting after punctuation
        if words[i - 1] and words[i - 1][-1] in good_breaks:
            diff -= 10

        # If we found a valid split (both fit), only consider other valid splits
        # If no valid split found yet, consider any split
        if both_fit:
            if not found_valid_split or diff < best_diff:
                best_diff = diff
                best_split = i
                found_valid_split = True
        elif not found_valid_split:
            if diff < best_diff:
                best_diff = diff
                best_split = i

    line1 = " ".join(words[:best_split])
    line2 = " ".join(words[best_split:])

    lines = []
    if line1:
        lines.append(line1)  # Don't truncate - let human editor handle overflow
    if line2:
        lines.append(line2)  # Don't truncate - let human editor handle overflow
    return lines if lines else [text]

    # Text is too long - fit into max_lines, keeping all content
    lines = []
    current_line = []
    current_len = 0
    remaining_words = []

    for i, word in enumerate(words):
        word_len = len(word)
        space = 1 if current_line else 0

        if current_len + space + word_len <= max_chars_per_line:
            current_line.append(word)
            current_len += space + word_len
        else:
            if current_line:
                lines.append(" ".join(current_line))
            if len(lines) >= max_lines - 1:
                # Last line: include all remaining words
                remaining_words = words[i:]
                break
            current_line = [word]
            current_len = word_len

    # Add remaining content to last line
    if remaining_words:
        last_line = " ".join(remaining_words)
        lines.append(last_line)
    elif current_line:
        lines.append(" ".join(current_line))

    return lines


def reoptimize_translated_segment(
    segment: dict,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
) -> dict:
    """
    Re-optimize a translated segment by rebalancing lines.
    Preserves timestamps but updates text and lines.

    Args:
        segment: Dict with start, end, text (and optionally lines)
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum lines per cue

    Returns:
        Updated segment dict with balanced lines
    """
    text = segment.get("text", "")
    if not text:
        return segment

    lines = rebalance_lines(text, max_chars_per_line, max_lines)

    return {
        "start": segment.get("start"),
        "end": segment.get("end"),
        "text": "\n".join(lines),
        "lines": lines,
    }


def _is_sentence_end(text: str, abbreviations: Set[str]) -> bool:
    """
    Detect if text ends at a sentence boundary.

    Handles:
    - Standard punctuation: . ! ?
    - Abbreviations: Mr., Dr., etc.
    - Decimal numbers: 3.5, 10.25
    - Ellipsis: ...
    """
    text = text.rstrip()
    if not text:
        return False

    # Strip trailing closing punctuation/quotes before boundary checks.
    # Example: 'rightful place."' should still count as sentence end.
    boundary_text = re.sub(r"[\"'”’)\]\}»]+$", "", text)
    if not boundary_text:
        return False

    # Must end with sentence-ending punctuation
    if boundary_text[-1] not in ".!?":
        return False

    # Handle ellipsis (not a sentence end in middle of speech)
    if boundary_text.endswith("..."):
        return False

    # Extract last word
    words = boundary_text.split()
    if not words:
        return True

    last_word = words[-1]
    last_word_core = last_word.rstrip(".!?").lower()
    normalized_abbreviations = {
        re.sub(r"[\"'”’)\]\}»]+$", "", abbr).rstrip(".!?").lower()
        for abbr in abbreviations
    }

    # Check if last word is a known abbreviation
    if last_word_core in normalized_abbreviations:
        return False

    # Check for single uppercase letter abbreviation (e.g., "J." for middle initial)
    if re.match(r"^[A-Z]\.$", last_word):
        return False

    # Keep list marker style "1." from forcing sentence breaks.
    # Numeric sentence endings in normal prose (e.g., "... in 1963.") should remain valid.
    if re.match(r"^\d+\.$", last_word) and len(words) == 1:
        return False

    return True


def _extract_cue_text(cue: dict) -> str:
    """Extract text from a cue, handling both 'lines' and 'text' formats."""
    if cue.get("lines"):
        return " ".join(cue["lines"]).strip()
    return (cue.get("text") or "").replace("\n", " ").strip()


def _split_into_sentences(text: str, abbreviations: Set[str]) -> List[str]:
    """
    Split text into individual sentences.

    Handles sentence-ending punctuation while respecting abbreviations.
    """
    if not text:
        return []

    sentences = []
    current = []
    words = text.split()
    normalized_abbreviations = {
        re.sub(r"[\"'”’)\]\}»]+$", "", abbr).rstrip(".!?").lower()
        for abbr in abbreviations
    }

    for i, word in enumerate(words):
        current.append(word)

        # Handle closing quotes/parens after punctuation, e.g. 'world."' or 'question?)'
        boundary_word = re.sub(r"[\"'”’)\]\}»]+$", "", word)

        # Check if this word ends a sentence
        if boundary_word[-1:] in ".!?":
            # Skip if it's an abbreviation (normalize by stripping punctuation and lowercasing)
            word_base = boundary_word.rstrip(".!?").lower()
            if word_base in normalized_abbreviations:
                continue

            # Skip single uppercase initials like "T." in names (e.g., "Josiah T. Walls").
            if re.match(r"^[A-Z]\.$", boundary_word):
                continue

            # Ellipsis: split if next word starts with capital (new sentence)
            if boundary_word.endswith("..."):
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    # If next word starts uppercase, it's a new sentence
                    if next_word and next_word[0].isupper():
                        sentence = " ".join(current)
                        sentences.append(sentence)
                        current = []
                continue

            # Skip decimal numbers like "3." if followed by a digit
            if re.match(r"^\d+\.$", boundary_word) and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word and next_word[0].isdigit():
                    continue

            # This is a sentence end
            sentence = " ".join(current)
            sentences.append(sentence)
            current = []

    # Add any remaining text
    if current:
        sentences.append(" ".join(current))

    return sentences


def merge_cues_to_sentences(
    cues: List[dict],
    src_lang: str = "en",
) -> List[SentenceGroup]:
    """
    Merge consecutive subtitle cues into complete sentences for better translation.

    This function groups cues that belong to the same sentence, tracking timing
    information so translations can be distributed back to the original cue timings.

    IMPORTANT: Also splits multi-sentence cues to ensure each sentence is translated
    separately, avoiding NLLB truncation issues.

    Args:
        cues: List of cue dicts with keys: start, end, text/lines
        src_lang: Source language code for abbreviation handling

    Returns:
        List of SentenceGroup objects with timing metadata
    """
    if not cues:
        return []

    # Get abbreviations for source language, fallback to English
    abbreviations = ABBREVIATIONS.get(src_lang, ABBREVIATIONS.get("en", set()))

    sentence_groups: List[SentenceGroup] = []
    current_sentence_parts: List[str] = []
    current_cue_indices: List[int] = []

    for i, cue in enumerate(cues):
        cue_text = _extract_cue_text(cue)

        # Skip empty cues but don't break sentence
        if not cue_text:
            continue

        # Check if this cue contains multiple sentences
        sentences_in_cue = _split_into_sentences(cue_text, abbreviations)

        if len(sentences_in_cue) > 1:
            # This cue has multiple sentences - process each separately

            # First, finalize any pending sentence with the first part
            if current_sentence_parts:
                # Append the first sentence from this cue to complete the pending sentence
                combined_text = (
                    " ".join(current_sentence_parts) + " " + sentences_in_cue[0]
                )
                first_idx = current_cue_indices[0]

                group = SentenceGroup(
                    sentence_text=combined_text.strip(),
                    source_cue_indices=current_cue_indices + [i],
                    start_time=cues[first_idx].get("start", 0.0),
                    end_time=cues[i].get("end", 0.0),
                    is_complete=True,
                )
                sentence_groups.append(group)
                current_sentence_parts = []
                current_cue_indices = []
                sentences_in_cue = sentences_in_cue[1:]  # Remove first sentence

            # Create groups for remaining complete sentences in this cue
            for j, sentence in enumerate(sentences_in_cue):
                is_last = j == len(sentences_in_cue) - 1
                is_complete = _is_sentence_end(sentence, abbreviations)

                if is_complete or not is_last:
                    # Complete sentence - create its own group
                    group = SentenceGroup(
                        sentence_text=sentence,
                        source_cue_indices=[i],
                        start_time=cues[i].get("start", 0.0),
                        end_time=cues[i].get("end", 0.0),
                        is_complete=is_complete,
                    )
                    sentence_groups.append(group)
                else:
                    # Last incomplete sentence - start accumulating
                    current_sentence_parts = [sentence]
                    current_cue_indices = [i]
        else:
            # Single sentence or partial - use original logic
            current_sentence_parts.append(cue_text)
            current_cue_indices.append(i)

            combined_text = " ".join(current_sentence_parts)
            force_break = len(current_cue_indices) >= MAX_CUES_PER_SENTENCE

            if _is_sentence_end(combined_text, abbreviations) or force_break:
                first_idx = current_cue_indices[0]
                last_idx = current_cue_indices[-1]

                group = SentenceGroup(
                    sentence_text=combined_text,
                    source_cue_indices=current_cue_indices.copy(),
                    start_time=cues[first_idx].get("start", 0.0),
                    end_time=cues[last_idx].get("end", 0.0),
                    is_complete=not force_break,
                )
                sentence_groups.append(group)

                current_sentence_parts = []
                current_cue_indices = []

    # Handle trailing incomplete sentence
    if current_sentence_parts:
        first_idx = current_cue_indices[0]
        last_idx = current_cue_indices[-1]

        group = SentenceGroup(
            sentence_text=" ".join(current_sentence_parts),
            source_cue_indices=current_cue_indices,
            start_time=cues[first_idx].get("start", 0.0),
            end_time=cues[last_idx].get("end", 0.0),
            is_complete=False,
        )
        sentence_groups.append(group)

    return sentence_groups


def distribute_translation_to_cues(
    translated_sentence: str,
    sentence_group: SentenceGroup,
    original_cues: List[dict],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
) -> List[dict]:
    """
    Distribute translated sentence text back to cues with proportional timing.

    Words are distributed based on the duration proportion of each original cue,
    ensuring that longer cues get more translated text.

    Args:
        translated_sentence: The translated sentence text
        sentence_group: SentenceGroup with timing metadata
        original_cues: Original cue list for timing reference
        max_chars_per_line: Maximum characters per subtitle line
        max_lines: Maximum lines per cue

    Returns:
        List of cue dicts with start, end, text, lines
    """
    if not translated_sentence or not sentence_group.source_cue_indices:
        return []

    translated_sentence = translated_sentence.strip()
    words = translated_sentence.split()

    if not words:
        return []

    cue_indices = sentence_group.source_cue_indices
    num_cues = len(cue_indices)
    num_words = len(words)

    # Calculate duration proportion for each cue
    total_duration = sentence_group.end_time - sentence_group.start_time
    cue_proportions: List[Tuple[int, float, float, float]] = (
        []
    )  # (idx, start, end, proportion)

    for idx in cue_indices:
        cue = original_cues[idx]
        start = cue.get("start", 0.0)
        end = cue.get("end", 0.0)
        duration = max(0.001, end - start)

        if total_duration > 0:
            proportion = duration / total_duration
        else:
            proportion = 1.0 / num_cues

        cue_proportions.append((idx, start, end, proportion))

    # Distribute words proportionally based on duration, respecting char limits
    max_chars_per_cue = max_chars_per_line * max_lines
    result_cues: List[dict] = []
    word_idx = 0

    for i, (orig_idx, start, end, proportion) in enumerate(cue_proportions):
        if i == num_cues - 1:
            # Last cue gets all remaining words
            cue_words = words[word_idx:]
        else:
            # Calculate target word count for this cue
            target_words = max(1, round(num_words * proportion))
            # Don't exceed remaining words
            remaining = num_words - word_idx
            target_words = min(target_words, remaining - (num_cues - i - 1))
            target_words = max(1, target_words)

            cue_words = words[word_idx : word_idx + target_words]
            word_idx += target_words

        # Build cue text and balance into lines
        cue_text = " ".join(cue_words)

        # If text exceeds max chars, split into multiple cues
        if len(cue_text) > max_chars_per_cue:
            # Split this cue's words into smaller groups
            duration = end - start
            sub_cues = _split_words_to_cues(
                cue_words,
                start,
                duration,
                max_chars_per_cue,
                max_chars_per_line,
                max_lines,
            )
            result_cues.extend(sub_cues)
        else:
            lines = rebalance_lines(cue_text, max_chars_per_line, max_lines)
            result_cues.append(
                {
                    "start": start,
                    "end": end,
                    "text": "\n".join(lines),
                    "lines": lines,
                }
            )

    return result_cues


def _split_words_to_cues(
    words: List[str],
    start_time: float,
    duration: float,
    max_chars_per_cue: int,
    max_chars_per_line: int,
    max_lines: int,
) -> List[dict]:
    """
    Split a list of words into multiple cues, each respecting max_chars_per_cue.

    Distributes timing proportionally (fallback when no alignment data available).
    """
    if not words:
        return []

    total_chars = sum(len(w) for w in words) + len(words) - 1  # +spaces
    cues = []
    current_words = []
    current_chars = 0
    chars_processed = 0

    for word in words:
        word_chars = len(word) + (1 if current_words else 0)  # +space if not first

        if current_chars + word_chars > max_chars_per_cue and current_words:
            # Finalize current cue
            cue_text = " ".join(current_words)
            lines = rebalance_lines(cue_text, max_chars_per_line, max_lines)

            # Calculate timing proportionally
            cue_start = start_time + (chars_processed / max(1, total_chars)) * duration
            chars_processed += current_chars
            cue_end = start_time + (chars_processed / max(1, total_chars)) * duration

            cues.append(
                {
                    "start": cue_start,
                    "end": cue_end,
                    "text": "\n".join(lines),
                    "lines": lines,
                }
            )

            current_words = [word]
            current_chars = len(word)
        else:
            current_words.append(word)
            current_chars += word_chars

    # Finalize last cue
    if current_words:
        cue_text = " ".join(current_words)
        lines = rebalance_lines(cue_text, max_chars_per_line, max_lines)

        cue_start = start_time + (chars_processed / max(1, total_chars)) * duration
        cue_end = start_time + duration  # Last cue ends at original end

        cues.append(
            {
                "start": cue_start,
                "end": cue_end,
                "text": "\n".join(lines),
                "lines": lines,
            }
        )

    return cues
