"""Replay cached verdicts from the one-time real-detector precompute run."""

import json

from ..config import CACHE_PATH
from .base import DetectorCacheMiss, DetectionResult


class ReplayDetector:
    def __init__(self, cache_path=CACHE_PATH):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._verdicts = {}
        for post_id_str, entry in payload["verdicts"].items():
            mapped = entry["mapped"]
            self._verdicts[int(post_id_str)] = DetectionResult(
                is_fake=mapped["is_fake"],
                confidence=mapped["confidence"],
                verdict_summary=mapped["verdict_summary"],
                provider=entry.get("provider", "replay"),
            )

    def detect(self, text, language, post_id):
        if post_id not in self._verdicts:
            raise DetectorCacheMiss(f"no cached verdict for post {post_id}")
        return self._verdicts[post_id]
