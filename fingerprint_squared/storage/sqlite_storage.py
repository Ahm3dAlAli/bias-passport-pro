"""
SQLite Storage Module

Provides persistent storage for evaluation experiments, probe responses,
and bias fingerprints. Enables historical analysis and comparison across
multiple evaluation runs.

Schema:
- experiments: Track evaluation runs
- probe_responses: Store individual probe responses
- fingerprints: Store aggregated fingerprints
- images: Track processed images
- comparisons: Store model comparison results
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fingerprint_squared.probes.social_inference_battery import ProbeResponse, ProbeType
from fingerprint_squared.core.bias_fingerprint import BiasFingerprint


@dataclass
class ExperimentRecord:
    """Record of a single evaluation experiment."""

    experiment_id: str
    model_id: str
    model_name: str
    dataset_name: str
    created_at: str
    completed_at: Optional[str] = None
    status: str = "running"  # running, completed, failed
    config: Dict[str, Any] = field(default_factory=dict)
    n_images: int = 0
    n_probes: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "config": self.config,
            "n_images": self.n_images,
            "n_probes": self.n_probes,
            "error": self.error,
        }


@dataclass
class ProbeResponseRecord:
    """Database record for a probe response."""

    response_id: str
    experiment_id: str
    image_id: str
    probe_type: str
    raw_response: str
    valence_score: Optional[float] = None
    stereotype_alignment: Optional[float] = None
    confidence_score: Optional[float] = None
    refusal: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0
    created_at: str = ""

    # Demographics
    gender: str = "unknown"
    age_range: str = "unknown"
    race: str = "unknown"


@dataclass
class FingerprintRecord:
    """Database record for a bias fingerprint."""

    fingerprint_id: str
    experiment_id: str
    model_id: str
    model_name: str
    overall_bias_score: float
    valence_bias: float
    stereotype_bias: float
    confidence_bias: float
    refusal_rate: float
    radar_dimensions: Dict[str, float]
    created_at: str
    fingerprint_json: str  # Full fingerprint as JSON


class SQLiteStorage:
    """
    SQLite-based storage for Fingerprint Squared experiments.

    Provides:
    - Persistent storage for experiments, responses, and fingerprints
    - Query interface for historical analysis
    - Export/import functionality
    - Comparison data storage

    Example:
        >>> storage = SQLiteStorage("./data/fingerprints.db")
        >>> exp_id = storage.create_experiment("gpt-4o", "GPT-4 Vision", "fhibe")
        >>> storage.save_responses(exp_id, responses)
        >>> storage.save_fingerprint(exp_id, fingerprint)
        >>> history = storage.get_model_history("gpt-4o")
    """

    SCHEMA_VERSION = 1

    CREATE_TABLES_SQL = """
    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );

    -- Experiments table
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT DEFAULT 'running',
        config_json TEXT,
        n_images INTEGER DEFAULT 0,
        n_probes INTEGER DEFAULT 0,
        error TEXT
    );

    -- Probe responses table
    CREATE TABLE IF NOT EXISTS probe_responses (
        response_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        image_id TEXT NOT NULL,
        probe_type TEXT NOT NULL,
        raw_response TEXT,
        valence_score REAL,
        stereotype_alignment REAL,
        confidence_score REAL,
        refusal INTEGER DEFAULT 0,
        error TEXT,
        latency_ms REAL DEFAULT 0,
        gender TEXT DEFAULT 'unknown',
        age_range TEXT DEFAULT 'unknown',
        race TEXT DEFAULT 'unknown',
        created_at TEXT NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    );

    -- Fingerprints table
    CREATE TABLE IF NOT EXISTS fingerprints (
        fingerprint_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        overall_bias_score REAL,
        valence_bias REAL,
        stereotype_bias REAL,
        confidence_bias REAL,
        refusal_rate REAL,
        radar_dimensions_json TEXT,
        fingerprint_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    );

    -- Images table (for tracking processed images)
    CREATE TABLE IF NOT EXISTS images (
        image_id TEXT PRIMARY KEY,
        image_path TEXT,
        gender TEXT,
        age_range TEXT,
        race TEXT,
        source TEXT,
        additional_attributes_json TEXT
    );

    -- Model comparisons table
    CREATE TABLE IF NOT EXISTS comparisons (
        comparison_id TEXT PRIMARY KEY,
        model_ids_json TEXT NOT NULL,
        comparison_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    -- Indices for common queries
    CREATE INDEX IF NOT EXISTS idx_responses_experiment ON probe_responses(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_responses_image ON probe_responses(image_id);
    CREATE INDEX IF NOT EXISTS idx_responses_probe ON probe_responses(probe_type);
    CREATE INDEX IF NOT EXISTS idx_fingerprints_model ON fingerprints(model_id);
    CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(model_id);
    CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
    """

    def __init__(self, db_path: str = "./data/fingerprints.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.CREATE_TABLES_SQL)

            # Check/update schema version
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,)
                )
            elif row[0] < self.SCHEMA_VERSION:
                self._migrate_schema(conn, row[0])

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with context management."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Migrate schema from older version."""
        # Add migration logic here as schema evolves
        pass

    # =========================================================================
    # Experiment Management
    # =========================================================================

    def create_experiment(
        self,
        model_id: str,
        model_name: str,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new experiment record.

        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            dataset_name: Name of the dataset being used
            config: Optional configuration dictionary

        Returns:
            experiment_id: Unique identifier for this experiment
        """
        import uuid

        experiment_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, model_id, model_name, dataset_name,
                    created_at, config_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    model_id,
                    model_name,
                    dataset_name,
                    created_at,
                    json.dumps(config or {}),
                    "running",
                ),
            )
            conn.commit()

        return experiment_id

    def update_experiment(
        self,
        experiment_id: str,
        status: Optional[str] = None,
        n_images: Optional[int] = None,
        n_probes: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update experiment record."""
        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "completed":
                updates.append("completed_at = ?")
                params.append(datetime.now().isoformat())

        if n_images is not None:
            updates.append("n_images = ?")
            params.append(n_images)

        if n_probes is not None:
            updates.append("n_probes = ?")
            params.append(n_probes)

        if error is not None:
            updates.append("error = ?")
            params.append(error)

        if not updates:
            return

        params.append(experiment_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?",
                params,
            )
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get experiment by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return ExperimentRecord(
                experiment_id=row["experiment_id"],
                model_id=row["model_id"],
                model_name=row["model_name"],
                dataset_name=row["dataset_name"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                status=row["status"],
                config=json.loads(row["config_json"] or "{}"),
                n_images=row["n_images"],
                n_probes=row["n_probes"],
                error=row["error"],
            )

    def list_experiments(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[ExperimentRecord]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            ExperimentRecord(
                experiment_id=row["experiment_id"],
                model_id=row["model_id"],
                model_name=row["model_name"],
                dataset_name=row["dataset_name"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                status=row["status"],
                config=json.loads(row["config_json"] or "{}"),
                n_images=row["n_images"],
                n_probes=row["n_probes"],
                error=row["error"],
            )
            for row in rows
        ]

    # =========================================================================
    # Probe Response Storage
    # =========================================================================

    def save_response(
        self,
        experiment_id: str,
        response: ProbeResponse,
        demographics: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a single probe response.

        Args:
            experiment_id: Parent experiment ID
            response: ProbeResponse to save
            demographics: Optional demographic info

        Returns:
            response_id
        """
        import uuid

        response_id = str(uuid.uuid4())[:12]
        created_at = datetime.now().isoformat()
        demographics = demographics or {}

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO probe_responses (
                    response_id, experiment_id, image_id, probe_type,
                    raw_response, valence_score, stereotype_alignment,
                    confidence_score, refusal, error, latency_ms,
                    gender, age_range, race, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response_id,
                    experiment_id,
                    response.image_id,
                    response.probe_type.value,
                    response.raw_response,
                    response.valence_score,
                    response.stereotype_alignment,
                    response.confidence_score,
                    1 if response.refusal else 0,
                    response.error,
                    response.latency_ms,
                    demographics.get("gender", "unknown"),
                    demographics.get("age_range", "unknown"),
                    demographics.get("race", "unknown"),
                    created_at,
                ),
            )
            conn.commit()

        return response_id

    def save_responses(
        self,
        experiment_id: str,
        responses: List[ProbeResponse],
        demographics_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> int:
        """
        Save multiple probe responses in batch.

        Args:
            experiment_id: Parent experiment ID
            responses: List of ProbeResponses
            demographics_map: Map of image_id to demographics

        Returns:
            Number of responses saved
        """
        import uuid

        demographics_map = demographics_map or {}
        created_at = datetime.now().isoformat()

        records = []
        for response in responses:
            demo = demographics_map.get(response.image_id, {})
            records.append((
                str(uuid.uuid4())[:12],
                experiment_id,
                response.image_id,
                response.probe_type.value,
                response.raw_response,
                response.valence_score,
                response.stereotype_alignment,
                response.confidence_score,
                1 if response.refusal else 0,
                response.error,
                response.latency_ms,
                demo.get("gender", "unknown"),
                demo.get("age_range", "unknown"),
                demo.get("race", "unknown"),
                created_at,
            ))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO probe_responses (
                    response_id, experiment_id, image_id, probe_type,
                    raw_response, valence_score, stereotype_alignment,
                    confidence_score, refusal, error, latency_ms,
                    gender, age_range, race, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

        return len(records)

    def get_responses(
        self,
        experiment_id: str,
        probe_type: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> List[ProbeResponseRecord]:
        """Get probe responses with optional filters."""
        query = "SELECT * FROM probe_responses WHERE experiment_id = ?"
        params = [experiment_id]

        if probe_type:
            query += " AND probe_type = ?"
            params.append(probe_type)

        if image_id:
            query += " AND image_id = ?"
            params.append(image_id)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            ProbeResponseRecord(
                response_id=row["response_id"],
                experiment_id=row["experiment_id"],
                image_id=row["image_id"],
                probe_type=row["probe_type"],
                raw_response=row["raw_response"],
                valence_score=row["valence_score"],
                stereotype_alignment=row["stereotype_alignment"],
                confidence_score=row["confidence_score"],
                refusal=bool(row["refusal"]),
                error=row["error"],
                latency_ms=row["latency_ms"],
                gender=row["gender"],
                age_range=row["age_range"],
                race=row["race"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # =========================================================================
    # Fingerprint Storage
    # =========================================================================

    def save_fingerprint(
        self,
        experiment_id: str,
        fingerprint: BiasFingerprint,
    ) -> str:
        """
        Save a bias fingerprint.

        Args:
            experiment_id: Parent experiment ID
            fingerprint: BiasFingerprint to save

        Returns:
            fingerprint_id
        """
        import uuid

        fingerprint_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO fingerprints (
                    fingerprint_id, experiment_id, model_id, model_name,
                    overall_bias_score, valence_bias, stereotype_bias,
                    confidence_bias, refusal_rate, radar_dimensions_json,
                    fingerprint_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fingerprint_id,
                    experiment_id,
                    fingerprint.model_id,
                    fingerprint.model_name,
                    fingerprint.overall_bias_score,
                    fingerprint.valence_bias,
                    fingerprint.stereotype_bias,
                    fingerprint.confidence_bias,
                    fingerprint.refusal_rate,
                    json.dumps(fingerprint.radar_dimensions),
                    fingerprint.to_json(),
                    created_at,
                ),
            )
            conn.commit()

        return fingerprint_id

    def get_fingerprint(self, fingerprint_id: str) -> Optional[BiasFingerprint]:
        """Get fingerprint by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT fingerprint_json FROM fingerprints WHERE fingerprint_id = ?",
                (fingerprint_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            data = json.loads(row["fingerprint_json"])
            return BiasFingerprint.load_from_dict(data) if hasattr(BiasFingerprint, 'load_from_dict') else None

    def get_latest_fingerprint(self, model_id: str) -> Optional[FingerprintRecord]:
        """Get most recent fingerprint for a model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM fingerprints
                WHERE model_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (model_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return FingerprintRecord(
                fingerprint_id=row["fingerprint_id"],
                experiment_id=row["experiment_id"],
                model_id=row["model_id"],
                model_name=row["model_name"],
                overall_bias_score=row["overall_bias_score"],
                valence_bias=row["valence_bias"],
                stereotype_bias=row["stereotype_bias"],
                confidence_bias=row["confidence_bias"],
                refusal_rate=row["refusal_rate"],
                radar_dimensions=json.loads(row["radar_dimensions_json"] or "{}"),
                created_at=row["created_at"],
                fingerprint_json=row["fingerprint_json"],
            )

    def get_model_history(
        self,
        model_id: str,
        limit: int = 10,
    ) -> List[FingerprintRecord]:
        """Get historical fingerprints for a model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM fingerprints
                WHERE model_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (model_id, limit),
            )
            rows = cursor.fetchall()

        return [
            FingerprintRecord(
                fingerprint_id=row["fingerprint_id"],
                experiment_id=row["experiment_id"],
                model_id=row["model_id"],
                model_name=row["model_name"],
                overall_bias_score=row["overall_bias_score"],
                valence_bias=row["valence_bias"],
                stereotype_bias=row["stereotype_bias"],
                confidence_bias=row["confidence_bias"],
                refusal_rate=row["refusal_rate"],
                radar_dimensions=json.loads(row["radar_dimensions_json"] or "{}"),
                created_at=row["created_at"],
                fingerprint_json=row["fingerprint_json"],
            )
            for row in rows
        ]

    # =========================================================================
    # Query Interface
    # =========================================================================

    def get_demographic_statistics(
        self,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """Get statistics grouped by demographics for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    gender, age_range, race,
                    COUNT(*) as n_responses,
                    AVG(valence_score) as avg_valence,
                    AVG(stereotype_alignment) as avg_stereotype,
                    AVG(confidence_score) as avg_confidence,
                    SUM(refusal) as n_refusals
                FROM probe_responses
                WHERE experiment_id = ?
                GROUP BY gender, age_range, race
                """,
                (experiment_id,),
            )
            rows = cursor.fetchall()

        stats = {}
        for row in rows:
            key = f"{row['gender']}_{row['age_range']}_{row['race']}"
            stats[key] = {
                "n_responses": row["n_responses"],
                "avg_valence": row["avg_valence"],
                "avg_stereotype": row["avg_stereotype"],
                "avg_confidence": row["avg_confidence"],
                "refusal_rate": row["n_refusals"] / row["n_responses"] if row["n_responses"] > 0 else 0,
            }

        return stats

    def get_probe_statistics(
        self,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """Get statistics grouped by probe type for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    probe_type,
                    COUNT(*) as n_responses,
                    AVG(valence_score) as avg_valence,
                    AVG(stereotype_alignment) as avg_stereotype,
                    AVG(confidence_score) as avg_confidence,
                    SUM(refusal) as n_refusals,
                    AVG(latency_ms) as avg_latency
                FROM probe_responses
                WHERE experiment_id = ?
                GROUP BY probe_type
                """,
                (experiment_id,),
            )
            rows = cursor.fetchall()

        stats = {}
        for row in rows:
            stats[row["probe_type"]] = {
                "n_responses": row["n_responses"],
                "avg_valence": row["avg_valence"],
                "avg_stereotype": row["avg_stereotype"],
                "avg_confidence": row["avg_confidence"],
                "refusal_rate": row["n_refusals"] / row["n_responses"] if row["n_responses"] > 0 else 0,
                "avg_latency_ms": row["avg_latency"],
            }

        return stats

    def compare_models(
        self,
        model_ids: List[str],
    ) -> Dict[str, Any]:
        """Compare latest fingerprints across multiple models."""
        comparison = {
            "models": {},
            "rankings": {},
        }

        for model_id in model_ids:
            fp = self.get_latest_fingerprint(model_id)
            if fp:
                comparison["models"][model_id] = {
                    "model_name": fp.model_name,
                    "overall_bias": fp.overall_bias_score,
                    "valence_bias": fp.valence_bias,
                    "stereotype_bias": fp.stereotype_bias,
                    "confidence_bias": fp.confidence_bias,
                    "refusal_rate": fp.refusal_rate,
                    "radar_dimensions": fp.radar_dimensions,
                    "created_at": fp.created_at,
                }

        # Compute rankings
        if comparison["models"]:
            sorted_by_bias = sorted(
                comparison["models"].items(),
                key=lambda x: x[1]["overall_bias"],
            )
            comparison["rankings"]["overall"] = [
                {"rank": i + 1, "model_id": model_id, "score": data["overall_bias"]}
                for i, (model_id, data) in enumerate(sorted_by_bias)
            ]

        return comparison

    # =========================================================================
    # Export/Import
    # =========================================================================

    def export_experiment(
        self,
        experiment_id: str,
        output_path: str,
    ) -> None:
        """Export experiment data to JSON."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        responses = self.get_responses(experiment_id)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT fingerprint_json FROM fingerprints WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
            fingerprint_json = row["fingerprint_json"] if row else None

        export_data = {
            "experiment": experiment.to_dict(),
            "responses": [
                {
                    "response_id": r.response_id,
                    "image_id": r.image_id,
                    "probe_type": r.probe_type,
                    "raw_response": r.raw_response,
                    "valence_score": r.valence_score,
                    "stereotype_alignment": r.stereotype_alignment,
                    "confidence_score": r.confidence_score,
                    "refusal": r.refusal,
                    "demographics": {
                        "gender": r.gender,
                        "age_range": r.age_range,
                        "race": r.race,
                    },
                }
                for r in responses
            ],
            "fingerprint": json.loads(fingerprint_json) if fingerprint_json else None,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def close(self) -> None:
        """Close any open resources."""
        pass  # Connections are closed after each operation
