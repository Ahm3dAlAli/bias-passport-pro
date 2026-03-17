"""
FastAPI Backend Server

REST API for Fingerprint Squared bias evaluation framework.

Endpoints:
- POST /api/evaluate - Run evaluation on a model
- GET /api/fingerprints/{model_id} - Get fingerprint for a model
- GET /api/fingerprints - List all fingerprints
- POST /api/compare - Compare multiple models
- GET /api/experiments - List experiments
- GET /api/experiments/{experiment_id} - Get experiment details
- POST /api/analyze-image - Analyze a single image
- GET /api/health - Health check
- GET /api/leaderboard - Get model leaderboard for dashboard
- WS /ws - WebSocket for real-time updates
- GET / - Dashboard HTML page
"""

from __future__ import annotations

import asyncio
import os
import json
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from contextlib import asynccontextmanager

from PIL import Image

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with: pip install fastapi uvicorn"
    )


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        # Clean up disconnected
        for conn in disconnected:
            self.active_connections.discard(conn)

    async def send_leaderboard_update(self, leaderboard: List[dict]):
        """Send leaderboard update to all clients."""
        await self.broadcast({
            "type": "leaderboard_update",
            "data": leaderboard,
            "timestamp": datetime.now().isoformat(),
        })

    async def send_evaluation_progress(self, model_id: str, progress: float, status: str):
        """Send evaluation progress update."""
        await self.broadcast({
            "type": "evaluation_progress",
            "model_id": model_id,
            "progress": progress,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        })


# Global connection manager
ws_manager = ConnectionManager()


# ============================================================================
# Pydantic Models
# ============================================================================

class EvaluationRequest(BaseModel):
    """Request to run a model evaluation."""

    model_id: str = Field(..., description="Model identifier (e.g., 'openrouter:gpt-4o')")
    model_name: Optional[str] = Field(None, description="Human-readable model name")
    dataset_path: Optional[str] = Field(None, description="Path to image dataset")
    n_images: int = Field(20, description="Number of images to evaluate")
    n_per_group: int = Field(5, description="Images per demographic group")
    probes: Optional[List[str]] = Field(None, description="Specific probes to run")
    seed: int = Field(42, description="Random seed for reproducibility")


class EvaluationResponse(BaseModel):
    """Response from evaluation request."""

    experiment_id: str
    status: str
    message: str
    fingerprint_id: Optional[str] = None


class FingerprintResponse(BaseModel):
    """Bias fingerprint response."""

    model_id: str
    model_name: str
    overall_bias_score: float
    valence_bias: float
    stereotype_bias: float
    confidence_bias: float
    refusal_rate: float
    radar_dimensions: Dict[str, float]
    created_at: str
    total_images: int
    total_probes: int


class CompareRequest(BaseModel):
    """Request to compare multiple models."""

    model_ids: List[str] = Field(..., description="List of model IDs to compare")


class CompareResponse(BaseModel):
    """Comparison response."""

    models: Dict[str, Dict[str, Any]]
    rankings: Dict[str, List[Dict[str, Any]]]


class ImageAnalysisRequest(BaseModel):
    """Request to analyze a single image."""

    model_id: str
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    probes: Optional[List[str]] = Field(None, description="Specific probes to run")


class ImageAnalysisResponse(BaseModel):
    """Response from single image analysis."""

    image_id: str
    model_id: str
    probe_responses: List[Dict[str, Any]]
    overall_valence: float
    overall_stereotype: float


class ExperimentSummary(BaseModel):
    """Summary of an experiment."""

    experiment_id: str
    model_id: str
    model_name: str
    dataset_name: str
    status: str
    created_at: str
    completed_at: Optional[str]
    n_images: int
    n_probes: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class LeaderboardEntry(BaseModel):
    """Entry in the model leaderboard."""

    rank: int
    model_id: str
    model_name: str
    overall_bias_score: float
    probe_scores: Dict[str, float]  # P1-P6 scores
    valence_bias: float
    stereotype_bias: float
    confidence_bias: float
    refusal_rate: float
    severity: str  # "low", "medium", "high"
    n_images: int
    last_updated: str


class LeaderboardResponse(BaseModel):
    """Full leaderboard response."""

    models: List[LeaderboardEntry]
    dataset_stats: Dict[str, Any]
    last_updated: str


# ============================================================================
# Application Factory
# ============================================================================

def create_app(
    storage_path: str = "./data/fingerprints.db",
    enable_cors: bool = True,
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        storage_path: Path to SQLite database
        enable_cors: Whether to enable CORS
        cors_origins: Allowed CORS origins

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Fingerprint² API",
        description="Bias fingerprinting for Vision-Language Models",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware
    if enable_cors:
        origins = cors_origins or [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Initialize storage
    from fingerprint_squared.storage.sqlite_storage import SQLiteStorage
    storage = SQLiteStorage(storage_path)
    app.state.storage = storage

    # Store running evaluations
    app.state.running_evaluations: Dict[str, asyncio.Task] = {}

    # ========================================================================
    # Health & Info Endpoints
    # ========================================================================

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp=datetime.now().isoformat(),
        )

    @app.get("/api/models")
    async def list_available_models():
        """List available VLM models."""
        from fingerprint_squared.models.openrouter_vlm import OPENROUTER_MODELS

        models = {
            "openrouter": list(OPENROUTER_MODELS.keys()),
            "local": [
                "qwen:Qwen2.5-VL-7B-Instruct",
                "qwen3:Qwen3-VL-8B-Instruct",
                "internvl:InternVL3-8B",
                "llama:Llama-3.2-11B-Vision-Instruct",
                "smol:SmolVLM-Instruct",
            ],
        }
        return models

    # ========================================================================
    # Evaluation Endpoints
    # ========================================================================

    @app.post("/api/evaluate", response_model=EvaluationResponse)
    async def start_evaluation(
        request: EvaluationRequest,
        background_tasks: BackgroundTasks,
    ):
        """
        Start a model evaluation.

        Runs asynchronously and returns immediately with experiment ID.
        Poll /api/experiments/{experiment_id} for status.
        """
        # Create experiment record
        experiment_id = storage.create_experiment(
            model_id=request.model_id,
            model_name=request.model_name or request.model_id,
            dataset_name=request.dataset_path or "synthetic",
            config={
                "n_images": request.n_images,
                "n_per_group": request.n_per_group,
                "probes": request.probes,
                "seed": request.seed,
            },
        )

        # Run evaluation in background
        async def run_evaluation():
            try:
                await _run_evaluation(
                    storage=storage,
                    experiment_id=experiment_id,
                    model_id=request.model_id,
                    model_name=request.model_name,
                    dataset_path=request.dataset_path,
                    n_images=request.n_images,
                    n_per_group=request.n_per_group,
                    probes=request.probes,
                    seed=request.seed,
                )
            except Exception as e:
                storage.update_experiment(
                    experiment_id,
                    status="failed",
                    error=str(e),
                )

        background_tasks.add_task(run_evaluation)

        return EvaluationResponse(
            experiment_id=experiment_id,
            status="started",
            message=f"Evaluation started. Poll /api/experiments/{experiment_id} for status.",
        )

    @app.get("/api/experiments", response_model=List[ExperimentSummary])
    async def list_experiments(
        model_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """List all experiments with optional filters."""
        experiments = storage.list_experiments(
            model_id=model_id,
            status=status,
            limit=limit,
        )

        return [
            ExperimentSummary(
                experiment_id=exp.experiment_id,
                model_id=exp.model_id,
                model_name=exp.model_name,
                dataset_name=exp.dataset_name,
                status=exp.status,
                created_at=exp.created_at,
                completed_at=exp.completed_at,
                n_images=exp.n_images,
                n_probes=exp.n_probes,
            )
            for exp in experiments
        ]

    @app.get("/api/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str):
        """Get experiment details including responses and fingerprint."""
        experiment = storage.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        responses = storage.get_responses(experiment_id)
        demo_stats = storage.get_demographic_statistics(experiment_id)
        probe_stats = storage.get_probe_statistics(experiment_id)

        return {
            "experiment": experiment.to_dict(),
            "statistics": {
                "by_demographic": demo_stats,
                "by_probe": probe_stats,
            },
            "n_responses": len(responses),
        }

    # ========================================================================
    # Fingerprint Endpoints
    # ========================================================================

    @app.get("/api/fingerprints")
    async def list_fingerprints(
        limit: int = Query(50, ge=1, le=500),
    ):
        """List all fingerprints."""
        # Get unique model IDs from experiments
        experiments = storage.list_experiments(status="completed", limit=limit)
        model_ids = list(set(exp.model_id for exp in experiments))

        fingerprints = []
        for model_id in model_ids:
            fp = storage.get_latest_fingerprint(model_id)
            if fp:
                fingerprints.append({
                    "fingerprint_id": fp.fingerprint_id,
                    "model_id": fp.model_id,
                    "model_name": fp.model_name,
                    "overall_bias_score": fp.overall_bias_score,
                    "valence_bias": fp.valence_bias,
                    "stereotype_bias": fp.stereotype_bias,
                    "confidence_bias": fp.confidence_bias,
                    "refusal_rate": fp.refusal_rate,
                    "radar_dimensions": fp.radar_dimensions,
                    "created_at": fp.created_at,
                })

        return fingerprints

    @app.get("/api/fingerprints/{model_id}", response_model=FingerprintResponse)
    async def get_fingerprint(model_id: str):
        """Get the latest fingerprint for a model."""
        fp = storage.get_latest_fingerprint(model_id)
        if not fp:
            raise HTTPException(
                status_code=404,
                detail=f"No fingerprint found for model: {model_id}"
            )

        # Get experiment for additional details
        exp = storage.get_experiment(fp.experiment_id)

        return FingerprintResponse(
            model_id=fp.model_id,
            model_name=fp.model_name,
            overall_bias_score=fp.overall_bias_score,
            valence_bias=fp.valence_bias,
            stereotype_bias=fp.stereotype_bias,
            confidence_bias=fp.confidence_bias,
            refusal_rate=fp.refusal_rate,
            radar_dimensions=fp.radar_dimensions,
            created_at=fp.created_at,
            total_images=exp.n_images if exp else 0,
            total_probes=exp.n_probes if exp else 0,
        )

    @app.get("/api/fingerprints/{model_id}/history")
    async def get_fingerprint_history(
        model_id: str,
        limit: int = Query(10, ge=1, le=100),
    ):
        """Get historical fingerprints for a model."""
        history = storage.get_model_history(model_id, limit=limit)
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No fingerprints found for model: {model_id}"
            )

        return [
            {
                "fingerprint_id": fp.fingerprint_id,
                "overall_bias_score": fp.overall_bias_score,
                "valence_bias": fp.valence_bias,
                "stereotype_bias": fp.stereotype_bias,
                "confidence_bias": fp.confidence_bias,
                "refusal_rate": fp.refusal_rate,
                "radar_dimensions": fp.radar_dimensions,
                "created_at": fp.created_at,
            }
            for fp in history
        ]

    # ========================================================================
    # Comparison Endpoints
    # ========================================================================

    @app.post("/api/compare", response_model=CompareResponse)
    async def compare_models(request: CompareRequest):
        """Compare multiple models."""
        if len(request.model_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 model IDs required for comparison"
            )

        comparison = storage.compare_models(request.model_ids)

        if not comparison.get("models"):
            raise HTTPException(
                status_code=404,
                detail="No fingerprints found for the specified models"
            )

        return CompareResponse(
            models=comparison["models"],
            rankings=comparison.get("rankings", {}),
        )

    # ========================================================================
    # Single Image Analysis
    # ========================================================================

    @app.post("/api/analyze-image", response_model=ImageAnalysisResponse)
    async def analyze_image(request: ImageAnalysisRequest):
        """Analyze a single image with specified model."""
        # Load image
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif request.image_url:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(request.image_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either image_base64 or image_url required"
            )

        # Create VLM
        from fingerprint_squared.models.openrouter_vlm import MultiProviderVLM
        try:
            vlm = MultiProviderVLM.create(request.model_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create VLM: {e}"
            )

        # Run probes
        from fingerprint_squared.probes.social_inference_battery import (
            SocialInferenceBattery,
            ProbeType,
        )
        battery = SocialInferenceBattery()

        # Filter probes if specified
        probe_types = list(ProbeType)
        if request.probes:
            probe_types = [
                pt for pt in ProbeType
                if pt.value in request.probes
            ]

        import uuid
        image_id = str(uuid.uuid4())[:8]

        # Run selected probes
        responses = []
        for probe_type in probe_types:
            response = await battery.run_single_probe(vlm, image, image_id, probe_type)
            responses.append(response)

        # Score responses
        from fingerprint_squared.scoring.llm_judge import LLMJudge
        judge = LLMJudge()

        probe_results = []
        total_valence = 0
        total_stereotype = 0
        n_scored = 0

        for response in responses:
            if not response.refusal and not response.error:
                scored = await judge.score_response(
                    response,
                    demographics={},
                    probe_question=battery.get_probe_prompt(response.probe_type),
                )
                if scored.valence_score is not None:
                    total_valence += scored.valence_score
                    total_stereotype += scored.stereotype_alignment or 0.5
                    n_scored += 1

                probe_results.append({
                    "probe_type": response.probe_type.value,
                    "response": response.raw_response,
                    "valence": scored.valence_score,
                    "stereotype": scored.stereotype_alignment,
                    "confidence": scored.confidence_score,
                    "refusal": response.refusal,
                })
            else:
                probe_results.append({
                    "probe_type": response.probe_type.value,
                    "response": response.raw_response,
                    "refusal": response.refusal,
                    "error": response.error,
                })

        # Cleanup
        if hasattr(vlm, 'close'):
            await vlm.close()

        return ImageAnalysisResponse(
            image_id=image_id,
            model_id=request.model_id,
            probe_responses=probe_results,
            overall_valence=total_valence / n_scored if n_scored > 0 else 0.0,
            overall_stereotype=total_stereotype / n_scored if n_scored > 0 else 0.5,
        )

    # ========================================================================
    # Export Endpoints
    # ========================================================================

    @app.get("/api/experiments/{experiment_id}/export")
    async def export_experiment(experiment_id: str):
        """Export experiment data as JSON."""
        experiment = storage.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Export to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            storage.export_experiment(experiment_id, f.name)
            return FileResponse(
                f.name,
                media_type="application/json",
                filename=f"experiment_{experiment_id}.json",
            )

    # ========================================================================
    # Leaderboard Endpoint
    # ========================================================================

    @app.get("/api/leaderboard", response_model=LeaderboardResponse)
    async def get_leaderboard():
        """Get model leaderboard sorted by overall bias score."""
        # Get all completed experiments
        experiments = storage.list_experiments(status="completed", limit=500)
        model_ids = list(set(exp.model_id for exp in experiments))

        entries = []
        for model_id in model_ids:
            fp = storage.get_latest_fingerprint(model_id)
            if not fp:
                continue

            # Get experiment for additional details
            exp = storage.get_experiment(fp.experiment_id)

            # Determine severity
            overall = fp.overall_bias_score
            if overall < 0.4:
                severity = "low"
            elif overall < 0.6:
                severity = "medium"
            else:
                severity = "high"

            # Map radar dimensions to P1-P6
            probe_scores = {
                "P1": fp.radar_dimensions.get("occupation", 0.0),
                "P2": fp.radar_dimensions.get("education", 0.0),
                "P3": fp.radar_dimensions.get("leadership", 0.0),
                "P4": fp.radar_dimensions.get("trustworthiness", 0.0),
                "P5": fp.radar_dimensions.get("lifestyle", 0.0),
                "P6": fp.radar_dimensions.get("neighborhood", 0.0),
            }

            entries.append(LeaderboardEntry(
                rank=0,  # Will be set after sorting
                model_id=fp.model_id,
                model_name=fp.model_name,
                overall_bias_score=fp.overall_bias_score,
                probe_scores=probe_scores,
                valence_bias=fp.valence_bias,
                stereotype_bias=fp.stereotype_bias,
                confidence_bias=fp.confidence_bias,
                refusal_rate=fp.refusal_rate,
                severity=severity,
                n_images=exp.n_images if exp else 0,
                last_updated=fp.created_at,
            ))

        # Sort by overall bias (lower is better)
        entries.sort(key=lambda e: e.overall_bias_score)

        # Assign ranks
        for i, entry in enumerate(entries, 1):
            entry.rank = i

        return LeaderboardResponse(
            models=entries,
            dataset_stats={
                "name": "FHIBE",
                "total_images": 10318,
                "jurisdictions": 81,
                "probes": 6,
                "dimensions": 3,
            },
            last_updated=datetime.now().isoformat(),
        )

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await ws_manager.connect(websocket)
        try:
            # Send initial leaderboard
            leaderboard = await get_leaderboard()
            await websocket.send_json({
                "type": "initial_leaderboard",
                "data": leaderboard.model_dump(),
                "timestamp": datetime.now().isoformat(),
            })

            # Keep connection alive and listen for messages
            while True:
                data = await websocket.receive_text()
                # Handle ping/pong for keepalive
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    # ========================================================================
    # Dashboard HTML Endpoint
    # ========================================================================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard HTML page."""
        dashboard_html = _generate_dashboard_html()
        return HTMLResponse(content=dashboard_html)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_alt():
        """Alternative dashboard route."""
        return await dashboard()

    # Mount static files if they exist
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def _generate_dashboard_html() -> str:
    """Generate the dashboard HTML with embedded JavaScript for real-time updates."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fingerprint² - VLM Bias Benchmark Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        .card-gradient {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            backdrop-filter: blur(10px);
        }
        .probe-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .probe-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        .severity-low { color: #22c55e; }
        .severity-medium { color: #eab308; }
        .severity-high { color: #ef4444; }
        .pulse-dot {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <!-- Header -->
    <header class="border-b border-slate-700/50 py-6">
        <div class="container mx-auto px-6">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-3xl font-bold">
                        Fingerprint<sup>2</sup>
                        <span class="text-cyan-400 ml-2 text-lg font-normal">VLM Bias Benchmark</span>
                    </h1>
                    <p class="text-slate-400 mt-1">Real-time bias evaluation across Vision-Language Models</p>
                </div>
                <div class="flex items-center space-x-4">
                    <div id="connection-status" class="flex items-center space-x-2">
                        <span class="pulse-dot w-2 h-2 bg-yellow-500 rounded-full"></span>
                        <span class="text-sm text-slate-400">Connecting...</span>
                    </div>
                    <button onclick="runBenchmark()" class="bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded-lg font-medium transition">
                        Run Benchmark
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Stats Bar -->
    <div class="border-b border-slate-700/50 py-4 bg-slate-900/30">
        <div class="container mx-auto px-6">
            <div class="grid grid-cols-4 gap-6 text-center">
                <div>
                    <div class="text-2xl font-bold text-cyan-400">10,318</div>
                    <div class="text-sm text-slate-400">Images (FHIBE)</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-purple-400">81</div>
                    <div class="text-sm text-slate-400">Jurisdictions</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-emerald-400">6</div>
                    <div class="text-sm text-slate-400">Probe Types</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-orange-400">3</div>
                    <div class="text-sm text-slate-400">Scoring Dimensions</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <main class="container mx-auto px-6 py-8">
        <!-- Probe Battery -->
        <section class="mb-12">
            <h2 class="text-xl font-semibold mb-6">Social Inference Battery (6 Probes)</h2>
            <div class="grid grid-cols-6 gap-4">
                <div class="probe-card card-gradient rounded-xl p-4 border border-slate-700/50">
                    <div class="text-cyan-400 font-bold mb-1">P1</div>
                    <div class="text-sm font-medium">Occupation</div>
                    <div class="text-xs text-slate-400 mt-1">Career inference</div>
                </div>
                <div class="probe-card card-gradient rounded-xl p-4 border border-slate-700/50">
                    <div class="text-purple-400 font-bold mb-1">P2</div>
                    <div class="text-sm font-medium">Education</div>
                    <div class="text-xs text-slate-400 mt-1">Attainment level</div>
                </div>
                <div class="probe-card card-gradient rounded-xl p-4 border border-slate-700/50">
                    <div class="text-emerald-400 font-bold mb-1">P3</div>
                    <div class="text-sm font-medium">Authority</div>
                    <div class="text-xs text-slate-400 mt-1">Leadership potential</div>
                </div>
                <div class="probe-card card-gradient rounded-xl p-4 border border-slate-700/50">
                    <div class="text-orange-400 font-bold mb-1">P4</div>
                    <div class="text-sm font-medium">Trustworthiness</div>
                    <div class="text-xs text-slate-400 mt-1">Reliability assessment</div>
                </div>
                <div class="probe-card card-gradient rounded-xl p-4 border border-slate-700/50">
                    <div class="text-pink-400 font-bold mb-1">P5</div>
                    <div class="text-sm font-medium">Lifestyle</div>
                    <div class="text-xs text-slate-400 mt-1">Socioeconomic status</div>
                </div>
                <div class="probe-card card-gradient rounded-xl p-4 border border-slate-700/50">
                    <div class="text-yellow-400 font-bold mb-1">P6</div>
                    <div class="text-sm font-medium">Neighborhood</div>
                    <div class="text-xs text-slate-400 mt-1">Geographic inference</div>
                </div>
            </div>
        </section>

        <!-- Leaderboard and Radar -->
        <div class="grid grid-cols-3 gap-8">
            <!-- Leaderboard -->
            <div class="col-span-2">
                <div class="card-gradient rounded-xl border border-slate-700/50 overflow-hidden">
                    <div class="px-6 py-4 border-b border-slate-700/50">
                        <h2 class="text-xl font-semibold">🏆 Bias Leaderboard</h2>
                        <p class="text-sm text-slate-400 mt-1">Lower scores indicate less bias (better)</p>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full">
                            <thead class="bg-slate-800/50">
                                <tr class="text-sm text-slate-400">
                                    <th class="px-4 py-3 text-left">Rank</th>
                                    <th class="px-4 py-3 text-left">Model</th>
                                    <th class="px-4 py-3 text-center">P1</th>
                                    <th class="px-4 py-3 text-center">P2</th>
                                    <th class="px-4 py-3 text-center">P3</th>
                                    <th class="px-4 py-3 text-center">P4</th>
                                    <th class="px-4 py-3 text-center">P5</th>
                                    <th class="px-4 py-3 text-center">P6</th>
                                    <th class="px-4 py-3 text-center">Overall</th>
                                    <th class="px-4 py-3 text-center">Severity</th>
                                </tr>
                            </thead>
                            <tbody id="leaderboard-body">
                                <tr class="text-center text-slate-500 py-8">
                                    <td colspan="10" class="py-8">
                                        <div class="flex flex-col items-center">
                                            <svg class="animate-spin h-8 w-8 text-cyan-500 mb-2" viewBox="0 0 24 24">
                                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            <span>Loading leaderboard...</span>
                                        </div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Radar Chart -->
            <div class="col-span-1">
                <div class="card-gradient rounded-xl border border-slate-700/50 p-6">
                    <h2 class="text-xl font-semibold mb-4">Bias Fingerprint</h2>
                    <canvas id="radarChart" width="300" height="300"></canvas>
                    <div class="mt-4 text-center">
                        <select id="model-select" class="bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm w-full" onchange="updateRadar()">
                            <option value="">Select a model</option>
                        </select>
                    </div>
                </div>

                <!-- Scoring Dimensions -->
                <div class="card-gradient rounded-xl border border-slate-700/50 p-6 mt-6">
                    <h3 class="font-semibold mb-4">Scoring Dimensions</h3>
                    <div class="space-y-3">
                        <div class="flex items-center">
                            <div class="w-3 h-3 bg-cyan-400 rounded-full mr-3"></div>
                            <span class="text-sm">Valence (positive/negative)</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-3 h-3 bg-purple-400 rounded-full mr-3"></div>
                            <span class="text-sm">Stereotype Alignment</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-3 h-3 bg-emerald-400 rounded-full mr-3"></div>
                            <span class="text-sm">Confidence Level</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-700/50 py-6 mt-12">
        <div class="container mx-auto px-6 text-center text-slate-400 text-sm">
            <p>Fingerprint² - Ethical AI Assessment Framework for Vision-Language Models</p>
            <p class="mt-1">Using FHIBE Dataset (10,318 images across 81 jurisdictions)</p>
        </div>
    </footer>

    <script>
        let ws;
        let radarChart;
        let leaderboardData = [];

        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                updateConnectionStatus('connected');
            };

            ws.onclose = () => {
                updateConnectionStatus('disconnected');
                // Reconnect after 3 seconds
                setTimeout(initWebSocket, 3000);
            };

            ws.onerror = () => {
                updateConnectionStatus('error');
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };

            // Keepalive
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send('ping');
                }
            }, 30000);
        }

        function updateConnectionStatus(status) {
            const statusEl = document.getElementById('connection-status');
            const dot = statusEl.querySelector('span:first-child');
            const text = statusEl.querySelector('span:last-child');

            if (status === 'connected') {
                dot.className = 'pulse-dot w-2 h-2 bg-green-500 rounded-full';
                text.textContent = 'Live';
            } else if (status === 'disconnected') {
                dot.className = 'pulse-dot w-2 h-2 bg-red-500 rounded-full';
                text.textContent = 'Reconnecting...';
            } else {
                dot.className = 'pulse-dot w-2 h-2 bg-yellow-500 rounded-full';
                text.textContent = 'Connecting...';
            }
        }

        function handleWebSocketMessage(message) {
            if (message.type === 'initial_leaderboard' || message.type === 'leaderboard_update') {
                const data = message.data;
                leaderboardData = data.models || data;
                updateLeaderboard(leaderboardData);
                updateModelSelect(leaderboardData);
            } else if (message.type === 'evaluation_progress') {
                // Show progress notification
                console.log('Evaluation progress:', message);
            }
        }

        function updateLeaderboard(models) {
            const tbody = document.getElementById('leaderboard-body');

            if (!models || models.length === 0) {
                tbody.innerHTML = `
                    <tr class="text-center text-slate-500">
                        <td colspan="10" class="py-8">
                            No models evaluated yet. Run a benchmark to get started.
                        </td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = models.map(model => {
                const severityClass = {
                    'low': 'severity-low',
                    'medium': 'severity-medium',
                    'high': 'severity-high'
                }[model.severity] || 'severity-medium';

                const severityLabel = model.severity.toUpperCase();

                return `
                    <tr class="border-t border-slate-700/30 hover:bg-slate-800/30 transition">
                        <td class="px-4 py-3 text-slate-400">#${model.rank}</td>
                        <td class="px-4 py-3 font-medium">${model.model_name}</td>
                        <td class="px-4 py-3 text-center">${formatScore(model.probe_scores?.P1)}</td>
                        <td class="px-4 py-3 text-center">${formatScore(model.probe_scores?.P2)}</td>
                        <td class="px-4 py-3 text-center">${formatScore(model.probe_scores?.P3)}</td>
                        <td class="px-4 py-3 text-center">${formatScore(model.probe_scores?.P4)}</td>
                        <td class="px-4 py-3 text-center">${formatScore(model.probe_scores?.P5)}</td>
                        <td class="px-4 py-3 text-center">${formatScore(model.probe_scores?.P6)}</td>
                        <td class="px-4 py-3 text-center font-bold">${formatScore(model.overall_bias_score)}</td>
                        <td class="px-4 py-3 text-center ${severityClass} font-medium">${severityLabel}</td>
                    </tr>
                `;
            }).join('');
        }

        function formatScore(score) {
            if (score === undefined || score === null) return '-';
            const s = parseFloat(score);
            let color = 'text-green-400';
            if (s >= 0.6) color = 'text-red-400';
            else if (s >= 0.4) color = 'text-yellow-400';
            return `<span class="${color}">${s.toFixed(2)}</span>`;
        }

        function updateModelSelect(models) {
            const select = document.getElementById('model-select');
            select.innerHTML = '<option value="">Select a model</option>' +
                models.map(m => `<option value="${m.model_id}">${m.model_name}</option>`).join('');

            if (models.length > 0 && !select.value) {
                select.value = models[0].model_id;
                updateRadar();
            }
        }

        function initRadarChart() {
            const ctx = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Occupation', 'Education', 'Authority', 'Trustworthiness', 'Lifestyle', 'Neighborhood'],
                    datasets: [{
                        label: 'Bias Score',
                        data: [0, 0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(6, 182, 212, 0.2)',
                        borderColor: 'rgba(6, 182, 212, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(6, 182, 212, 1)',
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 1,
                            grid: {
                                color: 'rgba(148, 163, 184, 0.2)'
                            },
                            angleLines: {
                                color: 'rgba(148, 163, 184, 0.2)'
                            },
                            pointLabels: {
                                color: '#94a3b8',
                                font: { size: 11 }
                            },
                            ticks: {
                                color: '#64748b',
                                backdropColor: 'transparent'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }

        function updateRadar() {
            const modelId = document.getElementById('model-select').value;
            const model = leaderboardData.find(m => m.model_id === modelId);

            if (model && radarChart) {
                const scores = model.probe_scores || {};
                radarChart.data.datasets[0].data = [
                    scores.P1 || 0,
                    scores.P2 || 0,
                    scores.P3 || 0,
                    scores.P4 || 0,
                    scores.P5 || 0,
                    scores.P6 || 0,
                ];
                radarChart.data.datasets[0].label = model.model_name;
                radarChart.update();
            }
        }

        async function runBenchmark() {
            const models = prompt('Enter model IDs (comma-separated):', 'gpt-4o,claude-3.5-sonnet');
            if (!models) return;

            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_id: models.split(',')[0].trim(),
                        n_images: 20
                    })
                });
                const result = await response.json();
                alert(`Evaluation started! Experiment ID: ${result.experiment_id}`);
            } catch (err) {
                alert('Error starting benchmark: ' + err.message);
            }
        }

        // Fallback: fetch leaderboard via REST if WebSocket fails
        async function fetchLeaderboard() {
            try {
                const response = await fetch('/api/leaderboard');
                const data = await response.json();
                leaderboardData = data.models;
                updateLeaderboard(leaderboardData);
                updateModelSelect(leaderboardData);
            } catch (err) {
                console.error('Failed to fetch leaderboard:', err);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initRadarChart();
            initWebSocket();
            // Fallback fetch after 2 seconds if WebSocket hasn't connected
            setTimeout(() => {
                if (!leaderboardData.length) {
                    fetchLeaderboard();
                }
            }, 2000);
        });
    </script>
</body>
</html>'''


# ============================================================================
# Background Task Implementation
# ============================================================================

async def _run_evaluation(
    storage,
    experiment_id: str,
    model_id: str,
    model_name: Optional[str],
    dataset_path: Optional[str],
    n_images: int,
    n_per_group: int,
    probes: Optional[List[str]],
    seed: int,
) -> None:
    """Run the actual evaluation (called in background)."""
    from fingerprint_squared.models.openrouter_vlm import MultiProviderVLM
    from fingerprint_squared.data.fhibe_loader import FHIBELoader, load_fhibe
    from fingerprint_squared.probes.social_inference_battery import (
        SocialInferenceBattery,
        ProbeType,
    )
    from fingerprint_squared.scoring.llm_judge import LLMJudge
    from fingerprint_squared.core.bias_fingerprint import FingerprintAggregator

    # Create VLM
    vlm = MultiProviderVLM.create(model_id)

    # Load or create dataset
    if dataset_path and Path(dataset_path).exists():
        dataset = load_fhibe(dataset_path)
    else:
        # Use synthetic dataset
        loader = FHIBELoader()
        dataset = loader.create_synthetic_dataset(n_per_intersection=n_per_group)

    # Get balanced sample
    sample = dataset.get_balanced_sample(n_per_group=n_per_group, seed=seed)

    # Initialize components
    battery = SocialInferenceBattery()
    judge = LLMJudge()
    aggregator = FingerprintAggregator()

    # Filter probes if specified
    probe_types = list(ProbeType)
    if probes:
        probe_types = [pt for pt in ProbeType if pt.value in probes]

    # Prepare images
    images = []
    demographics_map = {}
    for img in sample:
        try:
            pil_image = Image.open(img.image_path).convert("RGB")
            images.append((img.image_id, pil_image))
            demographics_map[img.image_id] = img.demographics
        except Exception:
            continue

    if not images:
        storage.update_experiment(
            experiment_id,
            status="failed",
            error="No valid images found",
        )
        return

    # Run probes
    all_responses = []
    for image_id, image in images:
        for probe_type in probe_types:
            response = await battery.run_single_probe(vlm, image, image_id, probe_type)
            all_responses.append(response)

    # Score responses
    probe_questions = {pt: battery.get_probe_prompt(pt) for pt in ProbeType}
    scored_responses = await judge.score_batch(
        all_responses,
        demographics_map,
        probe_questions,
    )

    # Add demographic info to responses
    for response in scored_responses:
        response.demographic_info = demographics_map.get(response.image_id, {})

    # Save responses
    storage.save_responses(experiment_id, scored_responses, demographics_map)

    # Generate fingerprint
    fingerprint = aggregator.aggregate(
        model_id=model_id,
        model_name=model_name or model_id,
        responses=scored_responses,
    )

    # Save fingerprint
    fingerprint_id = storage.save_fingerprint(experiment_id, fingerprint)

    # Update experiment status
    storage.update_experiment(
        experiment_id,
        status="completed",
        n_images=len(images),
        n_probes=len(all_responses),
    )

    # Broadcast update via WebSocket
    try:
        # Get updated leaderboard
        from fingerprint_squared.storage.sqlite_storage import SQLiteStorage
        experiments = storage.list_experiments(status="completed", limit=500)
        model_ids = list(set(exp.model_id for exp in experiments))

        leaderboard = []
        for mid in model_ids:
            fp = storage.get_latest_fingerprint(mid)
            if fp:
                overall = fp.overall_bias_score
                severity = "low" if overall < 0.4 else ("medium" if overall < 0.6 else "high")
                leaderboard.append({
                    "model_id": fp.model_id,
                    "model_name": fp.model_name,
                    "overall_bias_score": fp.overall_bias_score,
                    "probe_scores": {
                        "P1": fp.radar_dimensions.get("occupation", 0.0),
                        "P2": fp.radar_dimensions.get("education", 0.0),
                        "P3": fp.radar_dimensions.get("leadership", 0.0),
                        "P4": fp.radar_dimensions.get("trustworthiness", 0.0),
                        "P5": fp.radar_dimensions.get("lifestyle", 0.0),
                        "P6": fp.radar_dimensions.get("neighborhood", 0.0),
                    },
                    "severity": severity,
                })

        # Sort and add ranks
        leaderboard.sort(key=lambda x: x["overall_bias_score"])
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i

        await ws_manager.send_leaderboard_update(leaderboard)
    except Exception as e:
        # Don't fail the evaluation if WebSocket broadcast fails
        pass

    # Cleanup
    if hasattr(vlm, 'close'):
        await vlm.close()


# ============================================================================
# Default App Instance
# ============================================================================

app = create_app()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "fingerprint_squared.api.server:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
