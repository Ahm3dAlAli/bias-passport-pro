"""
Fingerprint²: A Comprehensive Ethical AI Assessment Framework for Vision-Language Models

This framework provides systematic evaluation of bias and fairness in VLMs across
multiple dimensions: demographic, intersectional, contextual, and representational.

Key Components:
    - SocialInferenceBattery: 6 probes for bias fingerprinting
    - LLMJudge: Automated scoring on valence, stereotype, confidence
    - BiasFingerprint: Unique bias profile for each model
    - FHIBELoader: Dataset loading for diverse facial images
    - BiasPassportPDF: Professional PDF report generation

Example:
    >>> from fingerprint_squared import FingerprintPipeline, load_fhibe
    >>> from fingerprint_squared.models import OpenAIVLM
    >>>
    >>> dataset = load_fhibe("path/to/images")
    >>> vlm = OpenAIVLM(model="gpt-4o")
    >>>
    >>> pipeline = FingerprintPipeline()
    >>> results = await pipeline.run(vlm, dataset, "gpt-4o", "GPT-4 Vision")
"""

__version__ = "0.1.0"
__author__ = "Fingerprint² Research Team"

# Core pipeline
from fingerprint_squared.core.fingerprint_pipeline import (
    FingerprintPipeline,
    MultiModelPipeline,
    PipelineConfig,
    fingerprint_model,
)

# Fingerprint aggregation
from fingerprint_squared.core.bias_fingerprint import (
    BiasFingerprint,
    FingerprintAggregator,
    FingerprintComparator,
)

# Social Inference Battery
from fingerprint_squared.probes.social_inference_battery import (
    SocialInferenceBattery,
    ProbeType,
    ProbeResponse,
)

# LLM Judge
from fingerprint_squared.scoring.llm_judge import LLMJudge, JudgeScores

# Dataset loading
from fingerprint_squared.data.fhibe_loader import (
    FHIBEDataset,
    FHIBEImage,
    FHIBELoader,
    load_fhibe,
    Gender,
    AgeRange,
    Race,
    SkinTone,
    FHIBE_JURISDICTIONS,
)

# Reporting
from fingerprint_squared.reporting.pdf_generator import (
    BiasPassportPDF,
    generate_passport,
)

# Analysis (new)
from fingerprint_squared.analysis.two_person import (
    TwoPersonAnalyzer,
    TwoPersonResult,
)

# Storage (new)
from fingerprint_squared.storage.sqlite_storage import SQLiteStorage

# Preprocessing (new)
from fingerprint_squared.preprocessing.image_processor import (
    ImagePreprocessor,
    BoundingBox,
    MaskingStrategy,
)

# Extended thinking judge (new)
from fingerprint_squared.scoring.llm_judge import ExtendedThinkingJudge

# Statistical tests (new)
from fingerprint_squared.metrics.statistical import StatisticalTests, StatisticalTestResult

# Legacy imports for backwards compatibility
try:
    from fingerprint_squared.core.evaluator import VLMEvaluator
    from fingerprint_squared.core.fingerprint import FingerprintGenerator
    from fingerprint_squared.core.pipeline import FingerprintSquared
    from fingerprint_squared.metrics.fairness import FairnessMetrics
    from fingerprint_squared.metrics.bias_scores import BiasScorer
    from fingerprint_squared.probes.bias_probes import BiasProbe
except ImportError:
    pass

__all__ = [
    # Core pipeline
    "FingerprintPipeline",
    "MultiModelPipeline",
    "PipelineConfig",
    "fingerprint_model",
    # Fingerprint
    "BiasFingerprint",
    "FingerprintAggregator",
    "FingerprintComparator",
    # Probes
    "SocialInferenceBattery",
    "ProbeType",
    "ProbeResponse",
    # Scoring
    "LLMJudge",
    "JudgeScores",
    "ExtendedThinkingJudge",
    # Data
    "FHIBEDataset",
    "FHIBEImage",
    "FHIBELoader",
    "load_fhibe",
    "Gender",
    "AgeRange",
    "Race",
    "SkinTone",
    "FHIBE_JURISDICTIONS",
    # Reporting
    "BiasPassportPDF",
    "generate_passport",
    # Analysis
    "TwoPersonAnalyzer",
    "TwoPersonResult",
    # Storage
    "SQLiteStorage",
    # Preprocessing
    "ImagePreprocessor",
    "BoundingBox",
    "MaskingStrategy",
    # Statistical tests
    "StatisticalTests",
    "StatisticalTestResult",
]
