"""Deterministic stub detector with configurable error rates.

Used for development, tests, and experiment sweeps. The runtime interface
is label-free: `detect(text, language, post_id)` resolves everything from
a table built once at construction time.
"""

import numpy as np

from ..config import STUB_FALSE_POSITIVE_RATE, STUB_SEED, STUB_SENSITIVITY
from .base import DetectionResult


class StubDetector:
    def __init__(
        self,
        posts,
        sensitivity=STUB_SENSITIVITY,
        false_positive_rate=STUB_FALSE_POSITIVE_RATE,
        seed=STUB_SEED,
    ):
        # Calibration only: the stub simulates a noisy sensor with the
        # configured error rates. The true labels are read HERE, once, and
        # never leave this constructor — detect() cannot see them.
        verdict_rng = np.random.default_rng(seed)
        confidence_rng = np.random.default_rng(seed + 1)
        self._verdicts = {}
        for post in posts:
            if post.true_label == "fake":
                is_fake = bool(verdict_rng.random() < sensitivity)
            else:
                is_fake = bool(verdict_rng.random() < false_positive_rate)
            if is_fake is True:
                verdict_summary = "REFUTED"
            elif is_fake is False:
                verdict_summary = "SUPPORTED"
            else:
                verdict_summary = "UNCERTAIN"
            confidence = float(0.55 + 0.4 * confidence_rng.random())
            self._verdicts[post.id] = DetectionResult(
                is_fake=is_fake,
                confidence=confidence,
                verdict_summary=verdict_summary,
                provider="stub",
            )

    def detect(self, text, language, post_id):
        # text and language are accepted for interface parity and never
        # inspected; the verdict depends only on the post id
        return self._verdicts[post_id]
