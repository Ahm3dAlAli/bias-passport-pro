"""Configuration management utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of values to override

    Returns:
        OmegaConf DictConfig object
    """
    default_config = get_default_config()

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                file_config = OmegaConf.create(yaml.safe_load(f))
            config = OmegaConf.merge(default_config, file_config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config = default_config

    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    return config


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(OmegaConf.to_container(config), f, default_flow_style=False)


def get_default_config() -> DictConfig:
    """Get default configuration."""
    return OmegaConf.create({
        "evaluation": {
            "batch_size": 16,
            "num_workers": 4,
            "max_samples": None,
            "seed": 42,
            "device": "cuda",
        },
        "models": {
            "default_provider": "openai",
            "timeout": 60,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
        "metrics": {
            "demographic_parity": True,
            "equalized_odds": True,
            "calibration": True,
            "intersectional": True,
            "representational": True,
            "contextual": True,
        },
        "bias_dimensions": {
            "gender": ["male", "female", "non-binary"],
            "race": ["white", "black", "asian", "hispanic", "indigenous", "mixed"],
            "age": ["child", "young_adult", "middle_aged", "elderly"],
            "disability": ["able-bodied", "physical_disability", "cognitive_disability"],
            "socioeconomic": ["low_income", "middle_income", "high_income"],
        },
        "probes": {
            "counterfactual": True,
            "stereotype_association": True,
            "representation_gap": True,
            "context_sensitivity": True,
            "harmful_content": True,
        },
        "output": {
            "save_raw_responses": True,
            "generate_visualizations": True,
            "export_formats": ["json", "csv", "html"],
        },
        "logging": {
            "level": "INFO",
            "wandb": {
                "enabled": False,
                "project": "fingerprint-squared",
                "entity": None,
            },
        },
    })


def get_api_keys() -> Dict[str, Optional[str]]:
    """Get API keys from environment variables."""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "replicate": os.getenv("REPLICATE_API_TOKEN"),
        "huggingface": os.getenv("HF_TOKEN"),
    }
