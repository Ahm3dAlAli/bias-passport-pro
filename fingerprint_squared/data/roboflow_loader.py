"""
Roboflow Dataset Loader for Fingerprint²

Load face/person datasets from Roboflow for bias evaluation.
Supports any Roboflow dataset with face images.

Usage:
    from fingerprint_squared.data.roboflow_loader import RoboflowLoader

    loader = RoboflowLoader(api_key="your-roboflow-key")
    dataset = loader.load_dataset("workspace/project", version=1)
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum
import tempfile

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

from PIL import Image
import requests


@dataclass
class RoboflowImage:
    """Single image from Roboflow dataset."""

    image_id: str
    image_path: str  # Local path after download
    image_url: str   # Original URL
    width: int
    height: int
    annotations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Demographic info (may be inferred or annotated)
    demographics: Dict[str, str] = field(default_factory=dict)

    @property
    def has_demographics(self) -> bool:
        return bool(self.demographics)


@dataclass
class RoboflowDataset:
    """Dataset loaded from Roboflow."""

    name: str
    workspace: str
    project: str
    version: int
    images: List[RoboflowImage]
    classes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[RoboflowImage]:
        return iter(self.images)

    def get_sample(self, n: int, seed: int = 42) -> "RoboflowDataset":
        """Get random sample of images."""
        random.seed(seed)
        sampled = random.sample(self.images, min(n, len(self.images)))
        return RoboflowDataset(
            name=self.name,
            workspace=self.workspace,
            project=self.project,
            version=self.version,
            images=sampled,
            classes=self.classes,
        )


class RoboflowLoader:
    """Load datasets from Roboflow for bias evaluation."""

    # Public face datasets on Roboflow that can be used
    PUBLIC_DATASETS = {
        "face-detection": "roboflow-100/faces-detection",
        "wider-face": "wider-face/face-detection",
        "face-mask": "face-mask-detection/face-mask-detection",
        "celebrity-faces": "celebrity-faces/celebrity-faces",
        "human-faces": "human-faces/human-face-detection",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Roboflow loader.

        Args:
            api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var)
            cache_dir: Directory to cache downloaded images
        """
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "roboflow_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not ROBOFLOW_AVAILABLE:
            print("Warning: roboflow package not installed. Install with: pip install roboflow")

        self._rf = None

    @property
    def rf(self):
        """Lazy load Roboflow client."""
        if self._rf is None:
            if not ROBOFLOW_AVAILABLE:
                raise ImportError("roboflow package required. Install with: pip install roboflow")
            if not self.api_key:
                raise ValueError("Roboflow API key required. Set ROBOFLOW_API_KEY or pass api_key parameter.")
            self._rf = Roboflow(api_key=self.api_key)
        return self._rf

    def list_public_datasets(self) -> Dict[str, str]:
        """List available public face datasets."""
        return self.PUBLIC_DATASETS.copy()

    def load_dataset(
        self,
        project_path: str,
        version: int = 1,
        max_images: Optional[int] = None,
        download: bool = True,
    ) -> RoboflowDataset:
        """
        Load a dataset from Roboflow.

        Args:
            project_path: Path like "workspace/project" or use PUBLIC_DATASETS key
            version: Dataset version number
            max_images: Limit number of images to load
            download: Whether to download images locally

        Returns:
            RoboflowDataset object
        """
        # Check if it's a shortcut name
        if project_path in self.PUBLIC_DATASETS:
            project_path = self.PUBLIC_DATASETS[project_path]

        # Parse workspace/project
        parts = project_path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid project path: {project_path}. Use 'workspace/project' format.")

        workspace, project = parts

        # Get project from Roboflow
        rf_project = self.rf.workspace(workspace).project(project)
        rf_dataset = rf_project.version(version)

        # Download dataset
        if download:
            dataset_dir = self.cache_dir / f"{workspace}_{project}_v{version}"
            if not dataset_dir.exists():
                print(f"Downloading dataset to {dataset_dir}...")
                rf_dataset.download("yolov8", location=str(dataset_dir))
        else:
            dataset_dir = None

        # Load images
        images = self._load_images_from_dataset(
            rf_dataset,
            dataset_dir,
            max_images,
        )

        return RoboflowDataset(
            name=f"{workspace}/{project}",
            workspace=workspace,
            project=project,
            version=version,
            images=images,
            classes=rf_project.classes if hasattr(rf_project, 'classes') else [],
        )

    def _load_images_from_dataset(
        self,
        rf_dataset,
        dataset_dir: Optional[Path],
        max_images: Optional[int],
    ) -> List[RoboflowImage]:
        """Load images from downloaded dataset."""
        images = []

        if dataset_dir and dataset_dir.exists():
            # Load from downloaded files
            for split in ["train", "valid", "test"]:
                split_dir = dataset_dir / split / "images"
                if split_dir.exists():
                    for img_path in split_dir.glob("*"):
                        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            try:
                                with Image.open(img_path) as img:
                                    width, height = img.size

                                image_id = hashlib.md5(str(img_path).encode()).hexdigest()[:12]

                                images.append(RoboflowImage(
                                    image_id=image_id,
                                    image_path=str(img_path),
                                    image_url="",
                                    width=width,
                                    height=height,
                                    metadata={"split": split},
                                ))

                                if max_images and len(images) >= max_images:
                                    return images

                            except Exception as e:
                                print(f"Error loading {img_path}: {e}")

        return images

    def load_from_url(
        self,
        image_urls: List[str],
        download: bool = True,
    ) -> RoboflowDataset:
        """
        Create dataset from a list of image URLs.

        Args:
            image_urls: List of image URLs
            download: Whether to download images locally

        Returns:
            RoboflowDataset object
        """
        images = []

        for url in image_urls:
            image_id = hashlib.md5(url.encode()).hexdigest()[:12]

            if download:
                # Download image
                local_path = self.cache_dir / f"{image_id}.jpg"
                if not local_path.exists():
                    try:
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        local_path.write_bytes(response.content)
                    except Exception as e:
                        print(f"Error downloading {url}: {e}")
                        continue

                try:
                    with Image.open(local_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Error reading {local_path}: {e}")
                    continue

                image_path = str(local_path)
            else:
                image_path = url
                width, height = 0, 0

            images.append(RoboflowImage(
                image_id=image_id,
                image_path=image_path,
                image_url=url,
                width=width,
                height=height,
            ))

        return RoboflowDataset(
            name="custom_urls",
            workspace="",
            project="",
            version=0,
            images=images,
        )

    def create_synthetic_dataset(self, n_images: int = 100) -> RoboflowDataset:
        """
        Create a synthetic dataset for testing (no API key needed).
        Uses placeholder images.

        Args:
            n_images: Number of synthetic images to create

        Returns:
            RoboflowDataset with synthetic images
        """
        images = []

        # Demographic combinations for synthetic data
        genders = ["male", "female"]
        ages = ["young", "middle", "senior"]

        for i in range(n_images):
            image_id = f"synthetic_{i:04d}"

            # Create a simple placeholder image
            img_path = self.cache_dir / f"{image_id}.png"
            if not img_path.exists():
                # Create colored placeholder
                color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255),
                )
                img = Image.new("RGB", (512, 512), color)
                img.save(img_path)

            # Assign random demographics
            demographics = {
                "gender": random.choice(genders),
                "age": random.choice(ages),
            }

            images.append(RoboflowImage(
                image_id=image_id,
                image_path=str(img_path),
                image_url="",
                width=512,
                height=512,
                demographics=demographics,
                metadata={"synthetic": True},
            ))

        return RoboflowDataset(
            name="synthetic",
            workspace="test",
            project="synthetic",
            version=1,
            images=images,
        )


def load_roboflow(
    project: str = "face-detection",
    version: int = 1,
    max_images: Optional[int] = None,
    api_key: Optional[str] = None,
) -> RoboflowDataset:
    """
    Convenience function to load a Roboflow dataset.

    Args:
        project: Project path or shortcut name (e.g., "face-detection")
        version: Dataset version
        max_images: Maximum images to load
        api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var)

    Returns:
        RoboflowDataset object
    """
    loader = RoboflowLoader(api_key=api_key)
    return loader.load_dataset(project, version=version, max_images=max_images)
