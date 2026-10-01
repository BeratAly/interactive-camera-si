"""Face recognition service (§7 second half, §35/§36/§37/§78).

Sits between the vision pipeline and the profile database:
  - maintains an in-memory index of stored embeddings (loaded at boot)
  - recognize(vector) -> (name, similarity) or None below threshold
  - enrollment sessions: collect N samples from the LIVE camera frames,
     quality-filter them, average-check consistency, then persist.

Ethics (§78): only compares against profiles the USER created on this PC.
Unknown faces stay UNKNOWN — there is no lookup path to any external source.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from the_machine.memory.database import Database
from the_machine.vision.embedder import FaceEmbedder, compare

logger = logging.getLogger("machine.vision.recognition")


@dataclass(frozen=True)
class MatchResult:
    name: str
    similarity: float          # cosine sim 0..1 — presented as likelihood (§38)


@dataclass
class EnrollmentSession:
    """Server-side state for a 'learn my face' flow driven by the UI."""
    profile_id: int
    name: str
    target_samples: int
    collected: list[np.ndarray] = field(default_factory=list)
    min_confidence: float = 0.7

    @property
    def progress(self) -> tuple[int, int]:
        return len(self.collected), self.target_samples

    def add_candidate(self, vector: np.ndarray, det_confidence: float) -> bool:
        """Accept one live detection. Returns True when stored as a sample."""
        if det_confidence < self.min_confidence:
            return False
        if len(self.collected) >= self.target_samples:
            return False
        # reject degenerate / duplicate samples (same frame twice)
        for prev in self.collected:
            if compare(prev, vector) > 0.995:
                return False
        self.collected.append(vector.astype(np.float32))
        return True


class FaceRecognitionService:
    MATCH_THRESHOLD = 0.42     # tuned for ArcFace cosine on aligned crops

    def __init__(self, db: Database, embedder: FaceEmbedder | None) -> None:
        self._db = db
        self._embedder = embedder
        self._lock = threading.Lock()
        self._index: dict[str, list[np.ndarray]] = {}   # name -> vectors
        self.reload_index()

    # ------------------------------------------------------------------ index
    def reload_index(self) -> None:
        try:
            rows = self._db.all_embeddings()
        except Exception:
            logger.exception("embedding index load failed")
            rows = []
        index: dict[str, list[np.ndarray]] = {}
        for _eid, name, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size:
                index.setdefault(name, []).append(vec)
        with self._lock:
            self._index = index
        logger.info("recognition index: %d identities, %d samples",
                    len(index), sum(len(v) for v in index.values()))

    @property
    def enabled(self) -> bool:
        return self._embedder is not None and self._embedder.available

    @property
    def known_names(self) -> list[str]:
        with self._lock:
            return sorted(self._index.keys())

    # -------------------------------------------------------------- matching
    def recognize(self, vector: np.ndarray) -> MatchResult | None:
        """Best-scoring identity over ALL stored samples; None below threshold."""
        best_name, best_sim = "", -1.0
        with self._lock:
            items = list(self._index.items())
        for name, vectors in items:
            for ref in vectors:
                if ref.shape != vector.shape:
                    continue
                sim = compare(ref, vector)
                if sim > best_sim:
                    best_name, best_sim = name, sim
        if best_name and best_sim >= self.MATCH_THRESHOLD:
            return MatchResult(best_name, round(float(best_sim), 4))
        return None

    # ------------------------------------------------------------ enrollment
    def start_enrollment(self, name: str, samples: int = 10) -> EnrollmentSession:
        clean = " ".join(name.split())[:60]
        if not clean:
            raise ValueError("name required")
        pid = self._db.create_profile(clean)
        return EnrollmentSession(profile_id=pid, name=clean,
                                 target_samples=max(3, min(samples, 30)))

    def finalize_enrollment(self, session: EnrollmentSession) -> int:
        """Persist collected vectors; returns saved sample count."""
        n = 0
        for vec in session.collected:
            self._db.add_embedding(session.profile_id,
                                   vec.astype(np.float32).tobytes(), 1.0)
            n += 1
        self.reload_index()
        logger.info("profile '%s' created with %d samples", session.name, n)
        return n

    def delete_profile(self, name: str) -> bool:
        prof = next((p for p in self._db.list_profiles()
                     if p.name.lower() == name.lower()), None)
        if prof is None:
            return False
        self._db.delete_profile(prof.id)
        self.reload_index()
        return True
