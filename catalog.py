"""
catalog.py - SHL Product Catalog loader and TF-IDF based retriever.

Downloads the catalog JSON once, caches it locally, and builds a TF-IDF
index for fast keyword/semantic retrieval. All URLs returned are from
the catalog; no invented URLs are ever produced.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
CACHE_FILE = Path(__file__).parent / "catalog_cache.json"
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours

# Map 'keys' field values → short type codes used in the response schema
KEY_TYPE_MAP = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def _derive_test_type(keys: list[str]) -> str:
    """Return the primary type code for an assessment, or 'K' as default."""
    for k in keys:
        if k in KEY_TYPE_MAP:
            return KEY_TYPE_MAP[k]
    return "K"


def _load_raw() -> list[dict]:
    """Fetch catalog from the remote URL (with local cache)."""
    if CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            logger.info("Using cached catalog (age=%.0fs).", age)
            with open(CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)

    logger.info("Downloading fresh catalog from %s …", CATALOG_URL)
    with httpx.Client(timeout=30) as client:
        resp = client.get(CATALOG_URL)
        resp.raise_for_status()
        # The catalog may contain embedded control characters; parse with strict=False
        import json as _json
        data = _json.loads(resp.content.decode("utf-8", errors="replace"), strict=False)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info("Catalog downloaded: %d items.", len(data))
    return data


def _build_text(item: dict) -> str:
    """Combine fields into a single searchable text blob."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return " ".join(p for p in parts if p).lower()


class CatalogRetriever:
    """
    Holds all catalog items and a TF-IDF index for retrieval.

    Usage:
        retriever = CatalogRetriever.load()
        results = retriever.search("java developer stakeholder communication", top_k=10)
    """

    def __init__(self, items: list[dict], vectorizer: TfidfVectorizer, matrix):
        self._items = items
        self._vectorizer = vectorizer
        self._matrix = matrix  # shape (n_docs, n_features)
        # Pre-build URL set for O(1) whitelist checks in the agent
        self.url_set: set[str] = {i["link"] for i in items}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "CatalogRetriever":
        raw = _load_raw()
        # Filter out items with no name or link
        items = [i for i in raw if i.get("name") and i.get("link")]

        texts = [_build_text(i) for i in items]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        logger.info("TF-IDF index built: %d docs × %d features.", *matrix.shape)
        return cls(items, vectorizer, matrix)

    def search(
        self,
        query: str,
        top_k: int = 10,
        job_level_filter: Optional[str] = None,
        key_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top_k catalog items ranked by cosine similarity to query.

        Uses multi-query expansion + name-match boost:
        - Multi-query expansion takes element-wise max across sub-queries so
          partial skill matches each contribute at full strength.
        - Name-match boost strongly up-ranks items whose NAME contains query
          keywords (e.g. 'Java 8' for a Java developer query).
        """
        scores = self._multi_query_scores(query)

        # Name-match boost — highest weight: title match is the strongest signal
        scores = scores + self._name_boost(query) * 0.40

        # Soft boosts — do not hard-exclude, just up-rank matching items
        if job_level_filter:
            scores = scores + self._level_boost(job_level_filter) * 0.20
        if key_filter:
            scores = scores + self._key_boost(key_filter) * 0.15

        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] < 1e-6:
                continue
            results.append(self._format(self._items[idx]))
        return results

    def _multi_query_scores(self, query: str) -> np.ndarray:
        """
        Split the query into up to 5 overlapping sub-queries and take the
        element-wise max of cosine similarity scores across all sub-queries.
        This prevents long compound queries from diluting individual term signals.
        """
        tokens = query.lower().split()
        if not tokens:
            return np.zeros(len(self._items))

        # Always include the full query
        sub_queries = [query.lower()]

        # Add individual content words (skip very short tokens)
        content_words = [t for t in tokens if len(t) > 3]
        if content_words:
            # Chunk into 3-word groups for bigram coverage
            for i in range(0, len(content_words), 2):
                chunk = " ".join(content_words[i:i+3])
                if chunk not in sub_queries:
                    sub_queries.append(chunk)

        # Cap at 5 sub-queries to stay fast
        sub_queries = sub_queries[:5]

        q_matrix = self._vectorizer.transform(sub_queries)
        sim_matrix = cosine_similarity(q_matrix, self._matrix)  # (n_queries, n_docs)
        # Element-wise max across all sub-queries
        return sim_matrix.max(axis=0)

    def _name_boost(self, query: str) -> np.ndarray:
        """
        Strong boost for items whose NAME contains one or more content words
        from the query. Each matching word adds 1.0 to the boost.
        E.g. 'Java 8 (New)' and 'Core Java' both get boosted for a Java query.
        This prevents generic description matches (like Ruby's 'developer') from
        outranking direct name matches.
        """
        # Extract meaningful words (len > 2, not stopwords)
        stopwords = {"the", "and", "for", "with", "that", "this", "are", "from",
                     "have", "has", "need", "who", "what", "how", "hiring",
                     "year", "years", "level", "developer", "engineer", "role"}
        tokens = [
            t.strip(".,;:").lower()
            for t in query.lower().split()
            if len(t) > 2 and t.lower() not in stopwords
        ]
        if not tokens:
            return np.zeros(len(self._items))

        boost = np.zeros(len(self._items))
        for i, item in enumerate(self._items):
            name_lower = item["name"].lower()
            # Count how many query tokens appear in the assessment name
            matches = sum(1 for t in tokens if t in name_lower)
            boost[i] = float(matches)
        return boost

    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact-ish lookup by assessment name (case-insensitive)."""
        name_lower = name.lower()
        for item in self._items:
            if item["name"].lower() == name_lower:
                return self._format(item)
        # fuzzy fallback: substring
        for item in self._items:
            if name_lower in item["name"].lower():
                return self._format(item)
        return None

    def get_all_names(self) -> list[str]:
        return [i["name"] for i in self._items]

    def item_count(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format(self, item: dict) -> dict:
        """Convert a raw catalog item dict to the standard output format."""
        return {
            "name": item["name"],
            "url": item["link"],   # raw JSON uses 'link', API uses 'url'
            "test_type": _derive_test_type(item.get("keys", [])),
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
            "keys": item.get("keys", []),
        }

    def _level_boost(self, level: str) -> np.ndarray:
        """Binary boost array: 1 where job_level matches, 0 otherwise."""
        level_lower = level.lower()
        boost = np.zeros(len(self._items))
        for i, item in enumerate(self._items):
            jls = [j.lower() for j in item.get("job_levels", [])]
            if any(level_lower in j for j in jls):
                boost[i] = 1.0
        return boost

    def _key_boost(self, key: str) -> np.ndarray:
        """Binary boost array: 1 where keys field contains the key."""
        key_lower = key.lower()
        boost = np.zeros(len(self._items))
        for i, item in enumerate(self._items):
            ks = [k.lower() for k in item.get("keys", [])]
            if any(key_lower in k for k in ks):
                boost[i] = 1.0
        return boost


# Singleton — loaded once at module import
_retriever: Optional[CatalogRetriever] = None


def get_retriever() -> CatalogRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever.load()
    return _retriever
