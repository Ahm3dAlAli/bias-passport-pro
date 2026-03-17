"""I/O utilities for loading and saving data."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonlines
import numpy as np
from PIL import Image


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with jsonlines.open(path, "r") as reader:
        for item in reader:
            data.append(item)
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save data to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(data)


def load_image(path: Union[str, Path]) -> Image.Image:
    """Load an image from path."""
    return Image.open(path).convert("RGB")


def load_images(
    paths: List[Union[str, Path]],
    resize: Optional[tuple[int, int]] = None,
) -> List[Image.Image]:
    """Load multiple images from paths."""
    images = []
    for path in paths:
        img = load_image(path)
        if resize:
            img = img.resize(resize, Image.LANCZOS)
        images.append(img)
    return images


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data))


def load_prompt_template(name: str) -> str:
    """Load a prompt template by name."""
    template_dir = Path(__file__).parent.parent.parent / "data" / "prompts"
    template_path = template_dir / f"{name}.txt"

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")

    with open(template_path, "r") as f:
        return f.read()


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)
