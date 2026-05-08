from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List


LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    label: str
    score: float
    backend: str


class SentimentAnalyzer:
    """
    Layered sentiment analyzer:
    1. HuggingFace Transformers with multiple candidate models
    2. Vietnamese-first lexical rules for short comments
    3. TextBlob fallback for English-like text
    """

    POSITIVE_TERMS = {
        "good", "great", "excellent", "love", "amazing", "nice", "happy", "best",
        "awesome", "cool", "recommend", "worth", "perfect", "impressive",
        "hay", "rat hay", "cuc hay", "qua hay", "hay qua", "hay lam", "hay nhat",
        "cuon", "cuon hut", "tot", "rat tot", "tot hon", "tuyet voi", "xuat sac",
        "an tuong", "hai long", "ung ho", "thich", "dang mua", "on", "bo ich",
        "chat luong", "uy tin", "tuyet", "dep", "de dung", "de hieu", "5 sao",
    }
    NEGATIVE_TERMS = {
        "bad", "terrible", "poor", "hate", "awful", "worst", "slow", "disappointed",
        "te", "kem", "that vong", "loi", "chan", "khong tot", "khong hay",
        "khong thich", "khong hai long", "qua te", "do", "lag", "rac", "phi",
        "ton thoi gian", "khong dang", "kem chat luong", "gian doi", "1 sao",
        "2 sao", "gia cao", "giao cham", "hong", "hu", "loang", "bat tien",
    }
    POSITIVE_HINTS = (
        "hay", "tot", "ok", "thich", "yeu", "cuon", "xuat sac", "tuyet voi",
        "hai long", "bo ich", "an tuong", "rat on", "dang mua", "uy tin",
    )
    NEGATIVE_HINTS = (
        "te", "kem", "chan", "loi", "that vong", "ton", "phi", "do", "lag",
        "rac", "giao cham", "khong on", "khong dang",
    )
    NEUTRAL_TERMS = {
        "binh thuong",
        "tam duoc",
        "cung duoc",
        "duoc",
        "on on",
        "khong co gi dac sac",
        "khong qua hay",
        "khong qua te",
        "trung binh",
        "vua phai",
        "tam on",
    }
    NEGATION_PREFIXES = ("khong", "chua", "chang", "cha", "khong he")

    def __init__(self, model_name: str, model_candidates: List[str] | None = None) -> None:
        self.model_name = model_name
        self.model_candidates = model_candidates or [model_name]
        if model_name not in self.model_candidates:
            self.model_candidates.insert(0, model_name)
        self._pipeline = None
        self._textblob_available = False
        self.backend_name = "lexicon-vn"
        self.loaded_model_name = ""
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        try:
            from transformers import pipeline

            for candidate in self.model_candidates:
                try:
                    self._pipeline = pipeline(
                        "text-classification",
                        model=candidate,
                        tokenizer=candidate,
                        truncation=True,
                    )
                    self.loaded_model_name = candidate
                    self.backend_name = f"transformers:{candidate}"
                    LOGGER.info("Loaded HuggingFace model: %s", candidate)
                    break
                except Exception as exc:
                    LOGGER.warning("Unable to load model '%s': %s", candidate, exc)
        except Exception as exc:
            LOGGER.warning("Transformers backend unavailable: %s", exc)

        try:
            from textblob import TextBlob  # noqa: F401

            self._textblob_available = True
        except Exception as exc:
            LOGGER.warning("TextBlob fallback unavailable: %s", exc)

    def analyze_comments(self, comments: List[str]) -> List[Dict[str, object]]:
        analyzed = []
        for comment in comments:
            prediction = self._predict(comment)
            analyzed.append(
                {
                    "comment": comment,
                    "sentiment": prediction.label,
                    "score": round(float(prediction.score), 4),
                    "backend": prediction.backend,
                }
            )
        return analyzed

    def _predict(self, text: str) -> PredictionResult:
        heuristic_prediction = self._predict_with_lexicon(text)
        if heuristic_prediction.backend == "lexicon-vn-neutral-override":
            return heuristic_prediction

        if heuristic_prediction.label != "Neutral":
            return heuristic_prediction

        if self._pipeline is not None:
            transformer_prediction = self._predict_with_transformers(text)
            if transformer_prediction.label != "Neutral" or transformer_prediction.score >= 0.6:
                return transformer_prediction

        if self._textblob_available and not self._looks_like_vietnamese(text):
            return self._predict_with_textblob(text)

        return heuristic_prediction

    def _predict_with_transformers(self, text: str) -> PredictionResult:
        prediction = self._pipeline(text, truncation=True, max_length=512)[0]
        normalized_label = self._normalize_label(prediction["label"], float(prediction["score"]))
        return PredictionResult(
            label=normalized_label,
            score=float(prediction["score"]),
            backend=self.backend_name,
        )

    def _predict_with_textblob(self, text: str) -> PredictionResult:
        from textblob import TextBlob

        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.12:
            return PredictionResult(label="Positive", score=polarity, backend="textblob")
        if polarity < -0.12:
            return PredictionResult(label="Negative", score=abs(polarity), backend="textblob")
        return PredictionResult(label="Neutral", score=0.5, backend="textblob")

    def _predict_with_lexicon(self, text: str) -> PredictionResult:
        normalized = self._normalize_text(text)
        if not normalized:
            return PredictionResult(label="Neutral", score=0.5, backend="lexicon-vn")

        if self._count_matches(normalized, self.NEUTRAL_TERMS) > 0:
            return PredictionResult(
                label="Neutral",
                score=0.72,
                backend="lexicon-vn-neutral-override",
            )

        positive_hits = self._count_matches(normalized, self.POSITIVE_TERMS)
        negative_hits = self._count_matches(normalized, self.NEGATIVE_TERMS)
        negated_positive_hits = self._count_negated_positive_terms(normalized)

        positive_hits -= negated_positive_hits
        negative_hits += negated_positive_hits

        if positive_hits > negative_hits:
            return PredictionResult(
                label="Positive",
                score=min(0.95, 0.58 + 0.09 * positive_hits),
                backend="lexicon-vn",
            )
        if negative_hits > positive_hits:
            return PredictionResult(
                label="Negative",
                score=min(0.95, 0.58 + 0.09 * negative_hits),
                backend="lexicon-vn",
            )

        short_text = len(normalized.split()) <= 9
        if short_text:
            if any(hint in normalized for hint in self.POSITIVE_HINTS):
                return PredictionResult(label="Positive", score=0.63, backend="lexicon-vn")
            if any(hint in normalized for hint in self.NEGATIVE_HINTS):
                return PredictionResult(label="Negative", score=0.63, backend="lexicon-vn")

        return PredictionResult(label="Neutral", score=0.5, backend="lexicon-vn")

    def _count_matches(self, normalized: str, terms: set[str]) -> int:
        return sum(self._contains_term(normalized, term) for term in terms)

    def _count_negated_positive_terms(self, normalized: str) -> int:
        hits = 0
        for prefix in self.NEGATION_PREFIXES:
            for term in self.POSITIVE_TERMS:
                if self._contains_term(normalized, f"{prefix} {term}"):
                    hits += 1
        return hits

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        return re.search(pattern, text) is not None

    @staticmethod
    def _looks_like_vietnamese(text: str) -> bool:
        lowered = text.lower()
        if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", lowered):
            return True
        padded = f" {SentimentAnalyzer._normalize_text(text)} "
        vietnamese_markers = (
            " khong ", " hay ", " phim ", " qua ", " rat ", " toi ", " ban ", " nha ", " voi ",
        )
        return any(marker in padded for marker in vietnamese_markers)

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower().strip()
        decomposed = unicodedata.normalize("NFD", lowered)
        without_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        without_accents = without_accents.replace("đ", "d").replace("Đ", "d")
        alphanumeric = re.sub(r"[^\w\s]", " ", without_accents)
        return re.sub(r"\s+", " ", alphanumeric).strip()

    @staticmethod
    def _normalize_label(label: str, score: float) -> str:
        raw = label.strip().lower()
        positive_labels = {"positive", "pos", "label_2", "label_4", "5 stars", "4 stars"}
        negative_labels = {"negative", "neg", "label_0", "1 star", "2 stars"}
        neutral_labels = {"neutral", "neu", "label_1", "label_3", "3 stars"}

        if raw in positive_labels:
            return "Positive"
        if raw in negative_labels:
            return "Negative"
        if raw in neutral_labels:
            return "Neutral"

        if "positive" in raw or raw.endswith("_pos"):
            return "Positive"
        if "negative" in raw or raw.endswith("_neg"):
            return "Negative"
        if "neutral" in raw or raw.endswith("_neu"):
            return "Neutral"

        if raw.startswith("label_"):
            try:
                numeric_label = int(raw.split("_")[-1])
            except ValueError:
                return "Positive" if score >= 0.7 else "Neutral"

            if numeric_label <= 1:
                return "Negative"
            if numeric_label == 2:
                return "Neutral"
            return "Positive"

        return "Positive" if score >= 0.7 else "Neutral"
