"""
FHIBE Dataset Loader

Loads and parses the Facial Histogram for Bias Evaluation (FHIBE) dataset
for use in bias fingerprinting experiments.

FHIBE provides diverse facial images with demographic annotations for
systematic bias testing across race, gender, and age intersections.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import hashlib


class Gender(Enum):
    """Gender categories following FHIBE self-reported pronouns."""
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNKNOWN = "unknown"


class AgeRange(Enum):
    """Age ranges for demographic grouping."""
    YOUNG = "18-30"
    MIDDLE = "31-50"
    SENIOR = "51+"
    UNKNOWN = "unknown"


class Race(Enum):
    """Race/ancestry categories aligned with FHIBE dataset."""
    WHITE = "white"
    BLACK = "black"
    ASIAN = "asian"
    HISPANIC = "hispanic"
    MIDDLE_EASTERN = "middle_eastern"
    SOUTH_ASIAN = "south_asian"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SkinTone(Enum):
    """Fitzpatrick skin tone scale (1-6) used in FHIBE."""
    TYPE_I = "1"      # Very light
    TYPE_II = "2"     # Light
    TYPE_III = "3"    # Medium light
    TYPE_IV = "4"     # Medium dark
    TYPE_V = "5"      # Dark
    TYPE_VI = "6"     # Very dark
    UNKNOWN = "unknown"


# FHIBE 81 Jurisdictions mapping (region -> countries)
FHIBE_JURISDICTIONS = {
    "Africa": [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo",
        "Cote d'Ivoire", "Democratic Republic of the Congo", "Djibouti",
        "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon",
        "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho",
        "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania",
        "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
        "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles",
        "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
        "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
    ],
    "Americas": [
        "Argentina", "Bolivia", "Brazil", "Canada", "Chile", "Colombia",
        "Costa Rica", "Cuba", "Dominican Republic", "Ecuador", "El Salvador",
        "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
        "Panama", "Paraguay", "Peru", "Puerto Rico", "Trinidad and Tobago",
        "United States", "Uruguay", "Venezuela"
    ],
    "Asia": [
        "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh",
        "Bhutan", "Brunei", "Cambodia", "China", "Cyprus", "Georgia",
        "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Israel",
        "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos",
        "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
        "North Korea", "Oman", "Pakistan", "Palestine", "Philippines",
        "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka",
        "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkey",
        "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"
    ],
    "Europe": [
        "Albania", "Andorra", "Austria", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
        "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
        "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway",
        "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia",
        "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine",
        "United Kingdom", "Vatican City"
    ],
    "Oceania": [
        "Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia",
        "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa",
        "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"
    ],
}


@dataclass
class FHIBEImage:
    """
    Single image from the FHIBE dataset.

    FHIBE contains 10,318 images across 81 jurisdictions with:
    - Self-reported pronouns (converted to gender)
    - Self-reported ancestry (mapped to race)
    - Annotated skin tone (Fitzpatrick scale)
    - Bounding boxes, keypoints, and segmentation masks
    """

    image_id: str
    image_path: str
    gender: Gender
    age_range: AgeRange
    race: Race

    # FHIBE-specific attributes
    jurisdiction: str = ""           # Country/region from 81 jurisdictions
    skin_tone: SkinTone = SkinTone.UNKNOWN
    subject_id: str = ""             # Subject identifier (1,981 unique subjects)

    # Annotation data
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    keypoints: Optional[List[Tuple[int, int]]] = None         # Facial keypoints
    has_segmentation_mask: bool = False

    # Additional metadata
    source: str = ""  # Original dataset source
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    additional_attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def demographic_key(self) -> str:
        """Unique key for this demographic intersection."""
        return f"{self.gender.value}_{self.age_range.value}_{self.race.value}"

    @property
    def jurisdiction_demographic_key(self) -> str:
        """Key including jurisdiction for geographic analysis."""
        return f"{self.jurisdiction}_{self.gender.value}_{self.race.value}"

    @property
    def demographics(self) -> Dict[str, str]:
        """Demographics as dict for probe usage."""
        return {
            "gender": self.gender.value,
            "age_range": self.age_range.value,
            "race": self.race.value,
            "jurisdiction": self.jurisdiction,
            "skin_tone": self.skin_tone.value,
        }

    @property
    def region(self) -> str:
        """Get geographic region for this jurisdiction."""
        for region, countries in FHIBE_JURISDICTIONS.items():
            if self.jurisdiction in countries:
                return region
        return "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "gender": self.gender.value,
            "age_range": self.age_range.value,
            "race": self.race.value,
            "jurisdiction": self.jurisdiction,
            "skin_tone": self.skin_tone.value,
            "subject_id": self.subject_id,
            "bounding_box": self.bounding_box,
            "keypoints": self.keypoints,
            "has_segmentation_mask": self.has_segmentation_mask,
            "source": self.source,
            "confidence_scores": self.confidence_scores,
            "additional_attributes": self.additional_attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FHIBEImage":
        """Create from dictionary."""
        return cls(
            image_id=data["image_id"],
            image_path=data["image_path"],
            gender=Gender(data.get("gender", "unknown")),
            age_range=AgeRange(data.get("age_range", "unknown")),
            race=Race(data.get("race", "unknown")),
            jurisdiction=data.get("jurisdiction", ""),
            skin_tone=SkinTone(data.get("skin_tone", "unknown")),
            subject_id=data.get("subject_id", ""),
            bounding_box=tuple(data["bounding_box"]) if data.get("bounding_box") else None,
            keypoints=data.get("keypoints"),
            has_segmentation_mask=data.get("has_segmentation_mask", False),
            source=data.get("source", ""),
            confidence_scores=data.get("confidence_scores", {}),
            additional_attributes=data.get("additional_attributes", {}),
        )


@dataclass
class FHIBEDataset:
    """
    FHIBE Dataset container.

    Provides balanced sampling across demographic intersections
    for systematic bias evaluation.

    FHIBE Dataset Statistics:
    - 10,318 total images
    - 1,981 unique subjects
    - 81 jurisdictions across 5 regions
    - Demographic attributes: pronouns, ancestry, skin tone
    """

    images: List[FHIBEImage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[FHIBEImage]:
        return iter(self.images)

    def __getitem__(self, idx: int) -> FHIBEImage:
        return self.images[idx]

    @property
    def demographic_distribution(self) -> Dict[str, int]:
        """Count images per demographic intersection."""
        distribution = {}
        for img in self.images:
            key = img.demographic_key
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    @property
    def jurisdiction_distribution(self) -> Dict[str, int]:
        """Count images per jurisdiction."""
        distribution = {}
        for img in self.images:
            key = img.jurisdiction
            if key:
                distribution[key] = distribution.get(key, 0) + 1
        return distribution

    @property
    def region_distribution(self) -> Dict[str, int]:
        """Count images per geographic region."""
        distribution = {}
        for img in self.images:
            region = img.region
            distribution[region] = distribution.get(region, 0) + 1
        return distribution

    @property
    def skin_tone_distribution(self) -> Dict[str, int]:
        """Count images per skin tone category."""
        distribution = {}
        for img in self.images:
            key = img.skin_tone.value
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    @property
    def unique_subjects(self) -> int:
        """Count unique subjects in dataset."""
        return len(set(img.subject_id for img in self.images if img.subject_id))

    @property
    def jurisdictions(self) -> List[str]:
        """List all unique jurisdictions."""
        return list(set(img.jurisdiction for img in self.images if img.jurisdiction))

    def filter_by_gender(self, gender: Gender) -> "FHIBEDataset":
        """Filter to specific gender."""
        filtered = [img for img in self.images if img.gender == gender]
        return FHIBEDataset(images=filtered, metadata=self.metadata)

    def filter_by_race(self, race: Race) -> "FHIBEDataset":
        """Filter to specific race."""
        filtered = [img for img in self.images if img.race == race]
        return FHIBEDataset(images=filtered, metadata=self.metadata)

    def filter_by_age(self, age_range: AgeRange) -> "FHIBEDataset":
        """Filter to specific age range."""
        filtered = [img for img in self.images if img.age_range == age_range]
        return FHIBEDataset(images=filtered, metadata=self.metadata)

    def filter_by_jurisdiction(self, jurisdiction: str) -> "FHIBEDataset":
        """Filter to specific jurisdiction."""
        filtered = [img for img in self.images if img.jurisdiction == jurisdiction]
        return FHIBEDataset(images=filtered, metadata=self.metadata)

    def filter_by_region(self, region: str) -> "FHIBEDataset":
        """Filter to specific geographic region."""
        filtered = [img for img in self.images if img.region == region]
        return FHIBEDataset(images=filtered, metadata=self.metadata)

    def filter_by_skin_tone(self, skin_tone: SkinTone) -> "FHIBEDataset":
        """Filter to specific skin tone."""
        filtered = [img for img in self.images if img.skin_tone == skin_tone]
        return FHIBEDataset(images=filtered, metadata=self.metadata)

    def get_balanced_sample(
        self,
        n_per_group: int = 10,
        groups: Optional[List[str]] = None,
        seed: int = 42,
    ) -> "FHIBEDataset":
        """
        Get balanced sample across demographic groups.

        Args:
            n_per_group: Number of images per demographic group
            groups: Specific demographic keys to include (None = all)
            seed: Random seed for reproducibility

        Returns:
            Balanced FHIBEDataset
        """
        random.seed(seed)

        # Group images by demographic
        by_demographic: Dict[str, List[FHIBEImage]] = {}
        for img in self.images:
            key = img.demographic_key
            if groups is None or key in groups:
                if key not in by_demographic:
                    by_demographic[key] = []
                by_demographic[key].append(img)

        # Sample from each group
        sampled = []
        for key, images in by_demographic.items():
            n = min(n_per_group, len(images))
            sampled.extend(random.sample(images, n))

        return FHIBEDataset(
            images=sampled,
            metadata={
                **self.metadata,
                "sampling": {
                    "n_per_group": n_per_group,
                    "seed": seed,
                    "groups": list(by_demographic.keys()),
                },
            },
        )

    def get_jurisdiction_balanced_sample(
        self,
        n_per_jurisdiction: int = 5,
        jurisdictions: Optional[List[str]] = None,
        seed: int = 42,
    ) -> "FHIBEDataset":
        """
        Get balanced sample across jurisdictions.

        Ensures geographic diversity in the sample.

        Args:
            n_per_jurisdiction: Number of images per jurisdiction
            jurisdictions: Specific jurisdictions to include (None = all)
            seed: Random seed for reproducibility

        Returns:
            Jurisdiction-balanced FHIBEDataset
        """
        random.seed(seed)

        # Group images by jurisdiction
        by_jurisdiction: Dict[str, List[FHIBEImage]] = {}
        for img in self.images:
            if not img.jurisdiction:
                continue
            if jurisdictions is None or img.jurisdiction in jurisdictions:
                if img.jurisdiction not in by_jurisdiction:
                    by_jurisdiction[img.jurisdiction] = []
                by_jurisdiction[img.jurisdiction].append(img)

        # Sample from each jurisdiction
        sampled = []
        for jurisdiction, images in by_jurisdiction.items():
            n = min(n_per_jurisdiction, len(images))
            sampled.extend(random.sample(images, n))

        return FHIBEDataset(
            images=sampled,
            metadata={
                **self.metadata,
                "sampling": {
                    "type": "jurisdiction_balanced",
                    "n_per_jurisdiction": n_per_jurisdiction,
                    "seed": seed,
                    "jurisdictions": list(by_jurisdiction.keys()),
                },
            },
        )

    def get_region_balanced_sample(
        self,
        n_per_region: int = 20,
        regions: Optional[List[str]] = None,
        seed: int = 42,
    ) -> "FHIBEDataset":
        """
        Get balanced sample across geographic regions.

        Args:
            n_per_region: Number of images per region
            regions: Specific regions (Africa, Americas, Asia, Europe, Oceania)
            seed: Random seed

        Returns:
            Region-balanced FHIBEDataset
        """
        random.seed(seed)

        # Group by region
        by_region: Dict[str, List[FHIBEImage]] = {}
        for img in self.images:
            region = img.region
            if regions is None or region in regions:
                if region not in by_region:
                    by_region[region] = []
                by_region[region].append(img)

        # Sample from each region
        sampled = []
        for region, images in by_region.items():
            n = min(n_per_region, len(images))
            sampled.extend(random.sample(images, n))

        return FHIBEDataset(
            images=sampled,
            metadata={
                **self.metadata,
                "sampling": {
                    "type": "region_balanced",
                    "n_per_region": n_per_region,
                    "seed": seed,
                    "regions": list(by_region.keys()),
                },
            },
        )

    def get_skin_tone_balanced_sample(
        self,
        n_per_tone: int = 20,
        seed: int = 42,
    ) -> "FHIBEDataset":
        """
        Get balanced sample across Fitzpatrick skin tone scale.

        Args:
            n_per_tone: Number of images per skin tone
            seed: Random seed

        Returns:
            Skin-tone balanced FHIBEDataset
        """
        random.seed(seed)

        # Group by skin tone
        by_tone: Dict[str, List[FHIBEImage]] = {}
        for img in self.images:
            if img.skin_tone != SkinTone.UNKNOWN:
                tone = img.skin_tone.value
                if tone not in by_tone:
                    by_tone[tone] = []
                by_tone[tone].append(img)

        # Sample from each tone
        sampled = []
        for tone, images in by_tone.items():
            n = min(n_per_tone, len(images))
            sampled.extend(random.sample(images, n))

        return FHIBEDataset(
            images=sampled,
            metadata={
                **self.metadata,
                "sampling": {
                    "type": "skin_tone_balanced",
                    "n_per_tone": n_per_tone,
                    "seed": seed,
                    "tones": list(by_tone.keys()),
                },
            },
        )

    def get_counterfactual_pairs(
        self,
        attribute: str = "gender",
        n_pairs: int = 50,
        seed: int = 42,
    ) -> List[Tuple[FHIBEImage, FHIBEImage]]:
        """
        Get pairs of images that differ only in specified attribute.

        This enables counterfactual fairness testing.

        Args:
            attribute: Which attribute to vary (gender, race, age_range)
            n_pairs: Number of pairs to return
            seed: Random seed

        Returns:
            List of (image_a, image_b) pairs
        """
        random.seed(seed)

        # Group by "everything except the target attribute"
        def get_matching_key(img: FHIBEImage) -> str:
            if attribute == "gender":
                return f"{img.age_range.value}_{img.race.value}"
            elif attribute == "race":
                return f"{img.gender.value}_{img.age_range.value}"
            elif attribute == "age_range":
                return f"{img.gender.value}_{img.race.value}"
            else:
                raise ValueError(f"Unknown attribute: {attribute}")

        groups: Dict[str, Dict[str, List[FHIBEImage]]] = {}
        for img in self.images:
            match_key = get_matching_key(img)
            attr_val = getattr(img, attribute).value

            if match_key not in groups:
                groups[match_key] = {}
            if attr_val not in groups[match_key]:
                groups[match_key][attr_val] = []
            groups[match_key][attr_val].append(img)

        # Find pairs
        pairs = []
        for match_key, attr_groups in groups.items():
            attr_values = list(attr_groups.keys())
            if len(attr_values) < 2:
                continue

            # Create pairs across attribute values
            for i, val_a in enumerate(attr_values):
                for val_b in attr_values[i + 1:]:
                    imgs_a = attr_groups[val_a]
                    imgs_b = attr_groups[val_b]

                    # Pair up images
                    for img_a, img_b in zip(imgs_a, imgs_b):
                        pairs.append((img_a, img_b))

        # Sample requested number
        if len(pairs) > n_pairs:
            pairs = random.sample(pairs, n_pairs)

        return pairs

    def save(self, path: str) -> None:
        """Save dataset to JSON."""
        data = {
            "metadata": self.metadata,
            "images": [img.to_dict() for img in self.images],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FHIBEDataset":
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)

        images = [FHIBEImage.from_dict(img) for img in data["images"]]
        return cls(images=images, metadata=data.get("metadata", {}))


class FHIBELoader:
    """
    Loader for FHIBE and compatible facial image datasets.

    Supports multiple source formats:
    - FHIBE native format (JSON + images)
    - UTKFace format (filename encodes demographics)
    - FairFace format (CSV annotations)
    - Custom directory structure

    Example:
        >>> loader = FHIBELoader()
        >>> dataset = loader.load_from_directory("path/to/images", format="utkface")
        >>> balanced = dataset.get_balanced_sample(n_per_group=20)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/fingerprint_squared")
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_from_directory(
        self,
        image_dir: str,
        format: str = "auto",
        annotation_file: Optional[str] = None,
    ) -> FHIBEDataset:
        """
        Load dataset from a directory of images.

        Args:
            image_dir: Path to directory containing images
            format: One of "auto", "fhibe", "utkface", "fairface", "custom"
            annotation_file: Path to annotation file (for fairface/custom formats)

        Returns:
            FHIBEDataset
        """
        image_dir = Path(image_dir)

        if format == "auto":
            format = self._detect_format(image_dir, annotation_file)

        if format == "fhibe_sony":
            return self._load_fhibe_sony(image_dir)
        elif format == "fhibe":
            return self._load_fhibe(image_dir)
        elif format == "utkface":
            return self._load_utkface(image_dir)
        elif format == "fairface":
            return self._load_fairface(image_dir, annotation_file)
        elif format == "custom":
            return self._load_custom(image_dir, annotation_file)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _detect_format(
        self,
        image_dir: Path,
        annotation_file: Optional[str],
    ) -> str:
        """Auto-detect dataset format."""
        # Check for Sony FHIBE native format (CSV in processed directory)
        fhibe_csv = image_dir / "data" / "processed" / "fhibe_face_crop_align" / "fhibe_face_crop_align.csv"
        if fhibe_csv.exists():
            return "fhibe_sony"

        # Check for FHIBE metadata JSON
        if (image_dir / "fhibe_metadata.json").exists():
            return "fhibe"

        # Check for FairFace CSV
        if annotation_file and annotation_file.endswith(".csv"):
            return "fairface"

        # Check filename pattern for UTKFace
        # UTKFace format: age_gender_race_date.jpg
        sample_files = list(image_dir.glob("*.jpg"))[:10]
        utkface_pattern = 0
        for f in sample_files:
            parts = f.stem.split("_")
            if len(parts) >= 3 and parts[0].isdigit():
                utkface_pattern += 1

        if utkface_pattern >= len(sample_files) * 0.7:
            return "utkface"

        return "custom"

    def _load_fhibe_sony(self, image_dir: Path) -> FHIBEDataset:
        """
        Load Sony FHIBE native format.

        Sony FHIBE dataset structure:
        - data/processed/fhibe_face_crop_align/fhibe_face_crop_align.csv
        - data/raw/fhibe_downsampled/{subject_id}/{session_id}/
          - faces_crop_and_align_{session_id}_{subject_id}.png (aligned face)

        CSV columns (actual structure):
        - image_id: UUID format like "session_uuid_subject_uuid"
        - filepath: relative path like "data/raw/fhibe_downsampled/..."
        - subject_id: UUID
        - pronoun: list format like "['1. He/him/his']"
        - ancestry: list format like "['11. Asia']"
        - nationality: list format like "['93. Indian']"
        - skin_type: Fitzpatrick 1-6
        """
        import csv
        import sys
        import ast

        # FHIBE CSV has very large fields (e.g., base64 encoded images)
        # Increase the field size limit to handle them
        csv.field_size_limit(sys.maxsize)

        csv_path = image_dir / "data" / "processed" / "fhibe_face_crop_align" / "fhibe_face_crop_align.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"FHIBE CSV not found at {csv_path}")

        # Mapping for pronouns to gender (handles numbered format like "1. He/him/his")
        def parse_pronoun(pronoun_raw: str) -> Gender:
            """Parse pronoun from format like \"['1. He/him/his']\" or plain string."""
            try:
                # Try to parse as Python list literal
                if pronoun_raw.startswith("["):
                    pronoun_list = ast.literal_eval(pronoun_raw)
                    if pronoun_list:
                        pronoun_raw = pronoun_list[0]

                # Extract the actual pronoun (remove number prefix like "1. ")
                pronoun = pronoun_raw.lower().strip()
                if ". " in pronoun:
                    pronoun = pronoun.split(". ", 1)[1]

                if "he" in pronoun:
                    return Gender.MALE
                elif "she" in pronoun:
                    return Gender.FEMALE
                elif "they" in pronoun:
                    return Gender.NON_BINARY
                else:
                    return Gender.UNKNOWN
            except (ValueError, SyntaxError, IndexError):
                return Gender.UNKNOWN

        # Parse nationality/jurisdiction from format like "['93. Indian']"
        def parse_nationality(nationality_raw: str) -> str:
            """Parse nationality from format like \"['93. Indian']\" or plain string."""
            try:
                if nationality_raw.startswith("["):
                    nationality_list = ast.literal_eval(nationality_raw)
                    if nationality_list:
                        nationality_raw = nationality_list[0]

                # Remove number prefix like "93. "
                nationality = nationality_raw.strip()
                if ". " in nationality:
                    nationality = nationality.split(". ", 1)[1]

                return nationality
            except (ValueError, SyntaxError, IndexError):
                return ""

        # Mapping for ancestry to race (handles format like "['11. Asia']")
        def parse_ancestry_to_race(ancestry_raw: str) -> Race:
            """Parse ancestry from format like \"['11. Asia']\" to Race enum."""
            try:
                if ancestry_raw.startswith("["):
                    ancestry_list = ast.literal_eval(ancestry_raw)
                    if ancestry_list:
                        ancestry_raw = ancestry_list[0]

                # Remove number prefix
                ancestry = ancestry_raw.lower().strip()
                if ". " in ancestry:
                    ancestry = ancestry.split(". ", 1)[1]

                if any(x in ancestry for x in ["europe", "white", "caucasian"]):
                    return Race.WHITE
                elif any(x in ancestry for x in ["africa", "black"]):
                    return Race.BLACK
                elif any(x in ancestry for x in ["asia", "asian", "east asia"]):
                    return Race.ASIAN
                elif any(x in ancestry for x in ["south asia", "indian", "india"]):
                    return Race.SOUTH_ASIAN
                elif any(x in ancestry for x in ["middle east", "arab"]):
                    return Race.MIDDLE_EASTERN
                elif any(x in ancestry for x in ["hispanic", "latin", "south america", "central america"]):
                    return Race.HISPANIC
                elif any(x in ancestry for x in ["mixed", "multi"]):
                    return Race.MIXED
                else:
                    return Race.UNKNOWN
            except (ValueError, SyntaxError, IndexError):
                return Race.UNKNOWN

        # Mapping for skin type to Fitzpatrick scale
        def map_skin_type(skin_type: str) -> SkinTone:
            try:
                val = str(int(float(skin_type)))
                return SkinTone(val)
            except (ValueError, TypeError):
                return SkinTone.UNKNOWN

        images = []
        jurisdictions_seen = set()
        skipped_count = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Get the image ID (use provided or generate from filepath)
                image_id = row.get("image_id", "")

                # Get the filepath directly from CSV
                filepath_rel = row.get("filepath", "").strip()

                if not filepath_rel:
                    skipped_count += 1
                    continue

                # Build absolute path: image_dir + filepath
                img_path = image_dir / filepath_rel

                if not img_path.exists():
                    skipped_count += 1
                    continue

                # Get subject_id
                subject_id = row.get("subject_id", "")

                # Parse pronouns to gender
                pronoun_raw = row.get("pronoun", "")
                gender = parse_pronoun(pronoun_raw)

                # Parse ancestry to race
                ancestry_raw = row.get("ancestry", "")
                race = parse_ancestry_to_race(ancestry_raw)

                # Parse jurisdiction from nationality or location_country
                nationality_raw = row.get("nationality", "") or row.get("location_country", "")
                jurisdiction = parse_nationality(nationality_raw)
                if jurisdiction:
                    jurisdictions_seen.add(jurisdiction)

                # Parse skin type
                skin_tone = map_skin_type(row.get("skin_type", ""))

                # Generate image_id if not provided
                if not image_id:
                    image_id = hashlib.md5(filepath_rel.encode()).hexdigest()[:16]

                img = FHIBEImage(
                    image_id=image_id,
                    image_path=str(img_path),
                    gender=gender,
                    age_range=AgeRange.UNKNOWN,  # Not provided in FHIBE
                    race=race,
                    jurisdiction=jurisdiction,
                    skin_tone=skin_tone,
                    subject_id=subject_id,
                    source="fhibe_sony",
                    additional_attributes={
                        "raw_pronoun": pronoun_raw,
                        "raw_ancestry": ancestry_raw,
                        "raw_nationality": nationality_raw,
                    },
                )
                images.append(img)

        print(f"[FHIBE Loader] Loaded {len(images)} images, skipped {skipped_count}")

        return FHIBEDataset(
            images=images,
            metadata={
                "source": "fhibe_sony",
                "path": str(image_dir),
                "total_images": len(images),
                "skipped_images": skipped_count,
                "unique_subjects": len(set(img.subject_id for img in images if img.subject_id)),
                "jurisdictions": list(jurisdictions_seen),
            },
        )

    def _load_fhibe(self, image_dir: Path) -> FHIBEDataset:
        """
        Load FHIBE native format.

        FHIBE dataset structure:
        - fhibe_metadata.json: Main metadata file
        - images/: Directory containing images
        - annotations/: Optional bounding boxes and keypoints

        Metadata fields:
        - id: Unique image identifier
        - filename: Image filename
        - subject_id: Subject identifier (1,981 unique)
        - jurisdiction: Country/region from 81 jurisdictions
        - pronouns: Self-reported (he/him, she/her, they/them)
        - ancestry: Self-reported ancestry
        - skin_tone: Fitzpatrick scale (1-6)
        - bounding_box: [x, y, width, height]
        - keypoints: Facial landmark points
        """
        metadata_path = image_dir / "fhibe_metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Mapping for pronouns to gender
        PRONOUNS_TO_GENDER = {
            "he/him": Gender.MALE,
            "she/her": Gender.FEMALE,
            "they/them": Gender.NON_BINARY,
            "he": Gender.MALE,
            "she": Gender.FEMALE,
            "they": Gender.NON_BINARY,
        }

        # Mapping for ancestry to race (simplified)
        def map_ancestry_to_race(ancestry: str) -> Race:
            ancestry = ancestry.lower()
            if any(x in ancestry for x in ["european", "white", "caucasian"]):
                return Race.WHITE
            elif any(x in ancestry for x in ["african", "black"]):
                return Race.BLACK
            elif any(x in ancestry for x in ["east asian", "chinese", "japanese", "korean"]):
                return Race.ASIAN
            elif any(x in ancestry for x in ["south asian", "indian", "pakistani", "bangladeshi"]):
                return Race.SOUTH_ASIAN
            elif any(x in ancestry for x in ["middle eastern", "arab", "persian"]):
                return Race.MIDDLE_EASTERN
            elif any(x in ancestry for x in ["hispanic", "latino", "latina", "latinx"]):
                return Race.HISPANIC
            elif any(x in ancestry for x in ["mixed", "multiracial"]):
                return Race.MIXED
            else:
                return Race.UNKNOWN

        images = []
        for entry in metadata.get("images", []):
            # Parse pronouns to gender
            pronouns = entry.get("pronouns", "").lower()
            gender = PRONOUNS_TO_GENDER.get(pronouns, Gender.UNKNOWN)

            # Parse ancestry to race
            ancestry = entry.get("ancestry", "")
            race = map_ancestry_to_race(ancestry)

            # Parse skin tone
            skin_tone_val = str(entry.get("skin_tone", "unknown"))
            try:
                skin_tone = SkinTone(skin_tone_val)
            except ValueError:
                skin_tone = SkinTone.UNKNOWN

            # Parse bounding box
            bbox = entry.get("bounding_box")
            if bbox and len(bbox) == 4:
                bbox = tuple(bbox)
            else:
                bbox = None

            # Build image path
            filename = entry.get("filename", "")
            if (image_dir / "images" / filename).exists():
                img_path = str(image_dir / "images" / filename)
            elif (image_dir / filename).exists():
                img_path = str(image_dir / filename)
            else:
                img_path = str(image_dir / filename)

            img = FHIBEImage(
                image_id=entry.get("id", hashlib.md5(filename.encode()).hexdigest()[:12]),
                image_path=img_path,
                gender=gender,
                age_range=AgeRange(entry.get("age_range", "unknown")),
                race=race,
                jurisdiction=entry.get("jurisdiction", ""),
                skin_tone=skin_tone,
                subject_id=entry.get("subject_id", ""),
                bounding_box=bbox,
                keypoints=entry.get("keypoints"),
                has_segmentation_mask=entry.get("has_segmentation_mask", False),
                source="fhibe",
                confidence_scores=entry.get("confidence", {}),
                additional_attributes={
                    "raw_ancestry": ancestry,
                    "raw_pronouns": pronouns,
                },
            )
            images.append(img)

        return FHIBEDataset(
            images=images,
            metadata={
                "source": "fhibe",
                "path": str(image_dir),
                "total_images": len(images),
                "unique_subjects": len(set(img.subject_id for img in images if img.subject_id)),
                "jurisdictions": list(set(img.jurisdiction for img in images if img.jurisdiction)),
            },
        )

    def _load_utkface(self, image_dir: Path) -> FHIBEDataset:
        """
        Load UTKFace format.

        Filename format: [age]_[gender]_[race]_[date&time].jpg
        - age: integer
        - gender: 0 (male), 1 (female)
        - race: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
        """
        GENDER_MAP = {0: Gender.MALE, 1: Gender.FEMALE}
        RACE_MAP = {
            0: Race.WHITE,
            1: Race.BLACK,
            2: Race.ASIAN,
            3: Race.SOUTH_ASIAN,  # Indian
            4: Race.MIXED,  # Others
        }

        def parse_age(age: int) -> AgeRange:
            if age <= 30:
                return AgeRange.YOUNG
            elif age <= 50:
                return AgeRange.MIDDLE
            else:
                return AgeRange.SENIOR

        images = []
        for img_path in image_dir.glob("*.jpg"):
            try:
                parts = img_path.stem.split("_")
                if len(parts) < 3:
                    continue

                age = int(parts[0])
                gender_code = int(parts[1])
                race_code = int(parts[2])

                img = FHIBEImage(
                    image_id=hashlib.md5(img_path.name.encode()).hexdigest()[:12],
                    image_path=str(img_path),
                    gender=GENDER_MAP.get(gender_code, Gender.UNKNOWN),
                    age_range=parse_age(age),
                    race=RACE_MAP.get(race_code, Race.UNKNOWN),
                    source="utkface",
                    additional_attributes={"raw_age": age},
                )
                images.append(img)

            except (ValueError, IndexError):
                continue

        return FHIBEDataset(
            images=images,
            metadata={"source": "utkface", "path": str(image_dir)},
        )

    def _load_fairface(
        self,
        image_dir: Path,
        annotation_file: Optional[str],
    ) -> FHIBEDataset:
        """
        Load FairFace format.

        Expects CSV with columns: file, age, gender, race, service_test
        """
        import csv

        if annotation_file is None:
            annotation_file = str(image_dir / "fairface_label_train.csv")

        RACE_MAP = {
            "White": Race.WHITE,
            "Black": Race.BLACK,
            "East Asian": Race.ASIAN,
            "Southeast Asian": Race.ASIAN,
            "Indian": Race.SOUTH_ASIAN,
            "Middle Eastern": Race.MIDDLE_EASTERN,
            "Latino_Hispanic": Race.HISPANIC,
        }

        AGE_MAP = {
            "0-2": AgeRange.YOUNG,
            "3-9": AgeRange.YOUNG,
            "10-19": AgeRange.YOUNG,
            "20-29": AgeRange.YOUNG,
            "30-39": AgeRange.MIDDLE,
            "40-49": AgeRange.MIDDLE,
            "50-59": AgeRange.SENIOR,
            "60-69": AgeRange.SENIOR,
            "more than 70": AgeRange.SENIOR,
        }

        images = []
        with open(annotation_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("file", "")
                img_path = image_dir / filename

                if not img_path.exists():
                    continue

                gender_str = row.get("gender", "").lower()
                gender = Gender.MALE if gender_str == "male" else (
                    Gender.FEMALE if gender_str == "female" else Gender.UNKNOWN
                )

                img = FHIBEImage(
                    image_id=hashlib.md5(filename.encode()).hexdigest()[:12],
                    image_path=str(img_path),
                    gender=gender,
                    age_range=AGE_MAP.get(row.get("age", ""), AgeRange.UNKNOWN),
                    race=RACE_MAP.get(row.get("race", ""), Race.UNKNOWN),
                    source="fairface",
                )
                images.append(img)

        return FHIBEDataset(
            images=images,
            metadata={"source": "fairface", "path": str(image_dir)},
        )

    def _load_custom(
        self,
        image_dir: Path,
        annotation_file: Optional[str],
    ) -> FHIBEDataset:
        """
        Load custom format with JSON annotations.

        Expects JSON file with structure:
        {
            "images": [
                {"filename": "...", "gender": "...", "age_range": "...", "race": "..."},
                ...
            ]
        }
        """
        if annotation_file is None:
            annotation_file = str(image_dir / "annotations.json")

        with open(annotation_file) as f:
            data = json.load(f)

        images = []
        for entry in data.get("images", []):
            filename = entry.get("filename", "")
            img_path = image_dir / filename

            if not img_path.exists():
                continue

            img = FHIBEImage(
                image_id=entry.get("id", hashlib.md5(filename.encode()).hexdigest()[:12]),
                image_path=str(img_path),
                gender=Gender(entry.get("gender", "unknown")),
                age_range=AgeRange(entry.get("age_range", "unknown")),
                race=Race(entry.get("race", "unknown")),
                source="custom",
                additional_attributes=entry.get("attributes", {}),
            )
            images.append(img)

        return FHIBEDataset(
            images=images,
            metadata={"source": "custom", "path": str(image_dir)},
        )

    def create_synthetic_dataset(
        self,
        n_per_intersection: int = 5,
        placeholder_image: str = "placeholder.jpg",
    ) -> FHIBEDataset:
        """
        Create a synthetic dataset for testing/development.

        Creates placeholder entries for all demographic intersections.
        Useful when real images aren't available.

        Args:
            n_per_intersection: Number of synthetic images per intersection
            placeholder_image: Path to placeholder image

        Returns:
            Synthetic FHIBEDataset
        """
        images = []
        idx = 0

        for gender in [Gender.MALE, Gender.FEMALE]:
            for age in [AgeRange.YOUNG, AgeRange.MIDDLE, AgeRange.SENIOR]:
                for race in [Race.WHITE, Race.BLACK, Race.ASIAN, Race.HISPANIC]:
                    for i in range(n_per_intersection):
                        img = FHIBEImage(
                            image_id=f"synthetic_{idx:05d}",
                            image_path=placeholder_image,
                            gender=gender,
                            age_range=age,
                            race=race,
                            source="synthetic",
                        )
                        images.append(img)
                        idx += 1

        return FHIBEDataset(
            images=images,
            metadata={
                "source": "synthetic",
                "n_per_intersection": n_per_intersection,
                "total_images": len(images),
            },
        )


# Convenience function for quick loading
def load_fhibe(
    path: str,
    format: str = "auto",
    annotation_file: Optional[str] = None,
) -> FHIBEDataset:
    """
    Quick-load function for FHIBE-compatible datasets.

    Example:
        >>> dataset = load_fhibe("path/to/images")
        >>> balanced = dataset.get_balanced_sample(n_per_group=20)
    """
    loader = FHIBELoader()
    return loader.load_from_directory(path, format=format, annotation_file=annotation_file)
