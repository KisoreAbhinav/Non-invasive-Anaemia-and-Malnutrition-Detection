"""Normalize short spoken answers into configured questionnaire values."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

LOGGER = logging.getLogger(__name__)

FUZZY_THRESHOLD = 85.0
SEMANTIC_THRESHOLD = 0.60
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODELS_PATH = Path(__file__).resolve().parents[2] / "models" / "sentence-transformers"
MIN_WIN_MARGIN = 3.0

_REPEAT_PHRASES = {
    "again",
    "can you repeat",
    "please repeat",
    "repeat",
    "repeat please",
    "say again",
}
_SKIP_PHRASES = {"skip", "skip questionnaire", "skip section"}


@dataclass(frozen=True)
class MatchResult:
    status: str
    transcription: str
    value: Any = None
    method: str | None = None
    confidence: float | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _normalize(text: str) -> str:
    lowered = text.casefold().replace("’", "'")
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", lowered)
    return " ".join(
        "no" if re.fullmatch(r"no+", token) else
        "yes" if re.fullmatch(r"yes+", token) else
        token
        for token in tokens
    )


def _candidate_phrases(option: Mapping[str, Any]) -> list[str]:
    values = [option.get("label"), option.get("value")]
    synonyms = option.get("synonyms", [])
    if isinstance(synonyms, list):
        values.extend(synonyms)
    return [normalized for item in values if (normalized := _normalize(str(item or "")))]


def _fuzzy_option_scores(question: Mapping[str, Any], text: str) -> list[tuple[float, Any]]:
    scores: list[tuple[float, Any]] = []
    text_tokens = set(text.split())
    for option in question.get("options", []):
        if not isinstance(option, Mapping) or "value" not in option:
            continue
        best = 0.0
        for candidate in _candidate_phrases(option):
            candidate_tokens = set(candidate.split())
            # Partial matching tiny tokens such as "yes"/"no" against a long
            # sentence produces dangerous false positives (e.g. matching a
            # few letters inside an unrelated word). Require a real token for
            # these fixed intents and reserve WRatio for meaningful phrases.
            if len(candidate_tokens) == 1 and len(candidate) <= 3:
                if candidate == text:
                    score = 100.0
                elif candidate in text_tokens:
                    # A configured longer phrase such as "no stamina" must be
                    # able to outweigh the isolated polarity word "no".
                    score = 93.0
                else:
                    score = float(fuzz.ratio(text, candidate))
            else:
                score = max(
                    float(fuzz.ratio(text, candidate)),
                    float(fuzz.WRatio(text, candidate)),
                    float(fuzz.token_set_ratio(text, candidate)),
                )
            if candidate == text:
                score = 100.0
            elif len(candidate_tokens) > 1 and candidate_tokens.issubset(text_tokens):
                score = max(score, 96.0)
            best = max(best, score)
        scores.append((best, option["value"]))
    return sorted(scores, key=lambda item: item[0], reverse=True)


class _LazySemanticMatcher:
    """Load sentence-transformers once, and only after fuzzy matching misses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._load_failed = False

    def _get_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    SEMANTIC_MODEL_NAME,
                    cache_folder=str(SEMANTIC_MODELS_PATH),
                    local_files_only=True,
                )
            except Exception:  # pragma: no cover - depends on model cache/network
                self._load_failed = True
                LOGGER.exception(
                    "Semantic answer model could not be loaded; unclear answers will be repeated"
                )
                return None
        return self._model

    def rank(self, question: Mapping[str, Any], text: str) -> list[tuple[float, Any]]:
        model = self._get_model()
        if model is None:
            return []

        option_values: list[Any] = []
        candidate_groups: list[list[str]] = []
        flat_candidates: list[str] = []
        for option in question.get("options", []):
            if not isinstance(option, Mapping) or "value" not in option:
                continue
            candidates = _candidate_phrases(option)
            if not candidates:
                continue
            option_values.append(option["value"])
            candidate_groups.append(candidates)
            flat_candidates.extend(candidates)

        if not flat_candidates:
            return []

        embeddings = model.encode(
            [text, *flat_candidates], convert_to_numpy=True, normalize_embeddings=True
        )
        text_embedding = embeddings[0]
        offset = 1
        ranked: list[tuple[float, Any]] = []
        for value, candidates in zip(option_values, candidate_groups, strict=True):
            candidate_embeddings = embeddings[offset : offset + len(candidates)]
            offset += len(candidates)
            best = max(float(text_embedding @ candidate) for candidate in candidate_embeddings)
            ranked.append((best, value))
        return sorted(ranked, key=lambda item: item[0], reverse=True)


_SEMANTIC_MATCHER = _LazySemanticMatcher()


_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_FILLERS = {
    "and",
    "age",
    "am",
    "i",
    "is",
    "months",
    "month",
    "my",
    "old",
    "years",
    "year",
}
_NUMBER_HOMOPHONES = {
    "ate": "eight",
    "for": "four",
    "to": "two",
    "too": "two",
    "won": "one",
}


def _normalize_number_homophones(text: str) -> str:
    tokens = text.split()
    has_age_context = bool(set(tokens) & {"month", "months", "old", "year", "years"})
    if len(tokens) == 1 or has_age_context:
        tokens = [_NUMBER_HOMOPHONES.get(token, token) for token in tokens]
    return " ".join(tokens)


def _parse_number_words(text: str) -> int | None:
    tokens = [token for token in text.replace("-", " ").split() if token not in _NUMBER_FILLERS]
    if not tokens or any(
        token not in _UNITS and token not in _TENS and token not in {"hundred", "thousand"}
        for token in tokens
    ):
        return None

    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token in _UNITS:
            current += _UNITS[token]
            saw_number = True
        elif token in _TENS:
            current += _TENS[token]
            saw_number = True
        elif token == "hundred":
            current = max(current, 1) * 100
            saw_number = True
        elif token == "thousand":
            total += max(current, 1) * 1000
            current = 0
            saw_number = True
    return total + current if saw_number else None


def parse_spoken_number(transcription: str) -> int | float | None:
    normalized = _normalize_number_homophones(_normalize(transcription))
    parsed_words = _parse_number_words(normalized)
    if parsed_words is not None:
        return parsed_words

    # STT often preserves the number but slightly distorts surrounding words
    # (for example, "four years old" -> "four years ago"). Recover the first
    # contiguous number phrase instead of rejecting the whole transcription.
    number_tokens = set(_UNITS) | set(_TENS) | {"hundred", "thousand"}
    run: list[str] = []
    for token in normalized.replace("-", " ").split():
        if token in number_tokens or token == "and" and run:
            run.append(token)
            continue
        if run:
            recovered = _parse_number_words(" ".join(run))
            if recovered is not None:
                return recovered
            run = []
    if run:
        recovered = _parse_number_words(" ".join(run))
        if recovered is not None:
            return recovered

    digit_match = re.search(r"(?<!\w)-?\d+(?:\.\d+)?", normalized)
    if digit_match is None:
        return None
    raw_number = digit_match.group(0)
    return float(raw_number) if "." in raw_number else int(raw_number)


def _number_result(question: Mapping[str, Any], transcription: str) -> MatchResult:
    value = parse_spoken_number(transcription)
    if value is None:
        return MatchResult(
            status="unclear",
            transcription=transcription,
            method="number",
            message="I could not understand that number. Please repeat it.",
        )

    minimum = question.get("min")
    maximum = question.get("max")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        return MatchResult(
            status="unclear",
            transcription=transcription,
            method="number",
            message=f"The answer must be between {minimum} and {maximum}. Please repeat it.",
        )
    return MatchResult(
        status="matched",
        transcription=transcription,
        value=value,
        method="number",
        confidence=1.0,
    )


def _confident_winner(
    ranked: list[tuple[float, Any]], threshold: float, margin: float
) -> tuple[float, Any] | None:
    if not ranked or ranked[0][0] <= threshold:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin:
        return None
    return ranked[0]


def match_answer(question: Mapping[str, Any], transcription: str) -> MatchResult:
    """Match one transcript using number, fuzzy, then semantic handling."""

    normalized = _normalize(transcription)
    if not normalized:
        return MatchResult(
            status="unclear",
            transcription=transcription,
            message="No speech was recognized. Please repeat the answer.",
        )
    if normalized in _REPEAT_PHRASES:
        return MatchResult(
            status="command",
            transcription=transcription,
            value="repeat",
            method="fixed_intent",
            confidence=1.0,
        )
    if normalized in _SKIP_PHRASES:
        return MatchResult(
            status="command",
            transcription=transcription,
            value="skip_section",
            method="fixed_intent",
            confidence=1.0,
        )

    if question.get("type") == "number":
        return _number_result(question, transcription)

    fuzzy_winner = _confident_winner(
        _fuzzy_option_scores(question, normalized), FUZZY_THRESHOLD, MIN_WIN_MARGIN
    )
    if fuzzy_winner is not None:
        confidence, value = fuzzy_winner
        return MatchResult(
            status="matched",
            transcription=transcription,
            value=value,
            method="fuzzy",
            confidence=round(confidence / 100.0, 4),
        )

    semantic_winner = _confident_winner(
        _SEMANTIC_MATCHER.rank(question, normalized), SEMANTIC_THRESHOLD, 0.03
    )
    if semantic_winner is not None:
        confidence, value = semantic_winner
        return MatchResult(
            status="matched",
            transcription=transcription,
            value=value,
            method="semantic",
            confidence=round(confidence, 4),
        )

    return MatchResult(
        status="unclear",
        transcription=transcription,
        message="I could not match that answer confidently. Please repeat it.",
    )
