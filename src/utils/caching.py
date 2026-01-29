"""Caching utilities for model responses."""

import sqlite3
import json
import hashlib
import os
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """SQLite-based cache for model responses."""

    def __init__(self, db_path: str = "outputs/cache/responses.db"):
        """Initialize cache.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                model_name TEXT,
                experiment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_experiment
            ON cache(model_name, experiment)
        """)
        conn.commit()
        conn.close()

    def make_key(
        self,
        item_id: str,
        model_name: str,
        prompt: str,
        generation_config: Dict[str, Any]
    ) -> str:
        """Generate unique cache key for an inference call.

        Args:
            item_id: Dataset item ID
            model_name: Model identifier
            prompt: Full prompt string
            generation_config: Generation parameters

        Returns:
            MD5 hash key
        """
        key_parts = [
            item_id,
            model_name,
            prompt,
            json.dumps(generation_config, sort_keys=True)
        ]
        key_string = "|".join(str(p) for p in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached response.

        Args:
            key: Cache key

        Returns:
            Cached value dict or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        result = conn.execute(
            "SELECT value FROM cache WHERE key = ?",
            (key,)
        ).fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
        return None

    def set(
        self,
        key: str,
        value: Dict,
        model_name: Optional[str] = None,
        experiment: Optional[str] = None
    ):
        """Store response in cache.

        Args:
            key: Cache key
            value: Value dict to store
            model_name: Model identifier for filtering
            experiment: Experiment name for filtering
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, value, model_name, experiment)
            VALUES (?, ?, ?, ?)
            """,
            (key, json.dumps(value), model_name, experiment)
        )
        conn.commit()
        conn.close()

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute(
            "SELECT 1 FROM cache WHERE key = ? LIMIT 1",
            (key,)
        ).fetchone()
        conn.close()
        return result is not None

    def delete(self, key: str):
        """Delete a cached entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        conn.close()

    def clear(self, model_name: Optional[str] = None, experiment: Optional[str] = None):
        """Clear cache entries.

        Args:
            model_name: If provided, only clear entries for this model
            experiment: If provided, only clear entries for this experiment
        """
        conn = sqlite3.connect(self.db_path)

        if model_name and experiment:
            conn.execute(
                "DELETE FROM cache WHERE model_name = ? AND experiment = ?",
                (model_name, experiment)
            )
        elif model_name:
            conn.execute("DELETE FROM cache WHERE model_name = ?", (model_name,))
        elif experiment:
            conn.execute("DELETE FROM cache WHERE experiment = ?", (experiment,))
        else:
            conn.execute("DELETE FROM cache")

        conn.commit()
        conn.close()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

        by_model = dict(conn.execute(
            "SELECT model_name, COUNT(*) FROM cache GROUP BY model_name"
        ).fetchall())

        by_experiment = dict(conn.execute(
            "SELECT experiment, COUNT(*) FROM cache GROUP BY experiment"
        ).fetchall())

        conn.close()

        return {
            "total_entries": total,
            "by_model": by_model,
            "by_experiment": by_experiment,
            "db_path": self.db_path
        }


class InMemoryCache:
    """Simple in-memory cache for fast access during runs."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, Dict] = {}

    def get(self, key: str) -> Optional[Dict]:
        return self._cache.get(key)

    def set(self, key: str, value: Dict):
        if len(self._cache) >= self.max_size:
            # Remove oldest 10%
            keys_to_remove = list(self._cache.keys())[:self.max_size // 10]
            for k in keys_to_remove:
                del self._cache[k]
        self._cache[key] = value

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
