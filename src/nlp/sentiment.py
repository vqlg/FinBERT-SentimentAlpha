import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)


def _resolve_device(pref: str):
    """Map device preference string to a value transformers pipeline accepts."""
    if pref == "auto":
        if torch.cuda.is_available():
            return 0           # first CUDA GPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return -1              # CPU
    if pref == "cpu":
        return -1
    if pref == "mps":
        return "mps"
    return 0  # assume cuda:0


class FinBERTScorer:
    """
    Scores financial headlines with ProsusAI/finbert.

    Output scale: confidence-weighted signed score in [-1, +1].
      positive label → +score
      negative label → -score
      neutral  label →  0 (near-neutral headlines don't move the needle)
    """

    _LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(self, model_name: str, batch_size: int, max_length: int, device: str):
        self._pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=_resolve_device(device),
            batch_size=batch_size,
            max_length=max_length,
            truncation=True,
        )
        logger.info("FinBERT loaded: %s", model_name)

    def score(self, texts: List[str]) -> np.ndarray:
        results = self._pipe(texts)
        return np.array([
            self._LABEL_SIGN.get(r["label"].lower(), 0.0) * r["score"]
            for r in results
        ])


class VADERScorer:
    """
    Lexical baseline using NLTK VADER.

    VADER captures punctuation intensity and capitalisation that transformer
    models miss (e.g. "EARNINGS MISS!" reads differently than "earnings miss").
    """

    def __init__(self):
        import nltk
        try:
            self._sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self._sia = SentimentIntensityAnalyzer()

    def score(self, texts: List[str]) -> np.ndarray:
        return np.array([self._sia.polarity_scores(t)["compound"] for t in texts])


class EnsembleSentimentScorer:
    """
    Weighted ensemble: FinBERT (deep semantic) + VADER (lexical heuristics).

    Default 75/25 weighting validated by research showing FinBERT dominates on
    clean financial text while VADER adds robustness on noisy headlines.
    """

    def __init__(self, cfg):
        self._finbert = FinBERTScorer(
            cfg.finbert_model, cfg.batch_size, cfg.max_length, cfg.device
        )
        self._vader = VADERScorer()
        self._fw = cfg.finbert_weight
        self._vw = cfg.vader_weight

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add finbert_score, vader_score, and ensemble sentiment_score to df.
        Input df must have a 'headline' column.
        """
        texts = df["headline"].tolist()
        fb = self._finbert.score(texts)
        vd = self._vader.score(texts)
        out = df.copy()
        out["finbert_score"] = fb
        out["vader_score"] = vd
        out["sentiment_score"] = self._fw * fb + self._vw * vd
        return out
