"""
Data loading and dataset utilities for Fingerprint Squared.

FHIBE Dataset:
- 10,318 images across 81 jurisdictions
- 1,981 unique subjects
- Demographic attributes: pronouns, ancestry, skin tone (Fitzpatrick scale)
- Annotations: bounding boxes, keypoints, segmentation masks
"""

from fingerprint_squared.data.fhibe_loader import (
    FHIBEDataset,
    FHIBEImage,
    FHIBELoader,
    Gender,
    AgeRange,
    Race,
    SkinTone,
    FHIBE_JURISDICTIONS,
    load_fhibe,
)

__all__ = [
    "FHIBEDataset",
    "FHIBEImage",
    "FHIBELoader",
    "Gender",
    "AgeRange",
    "Race",
    "SkinTone",
    "FHIBE_JURISDICTIONS",
    "load_fhibe",
]
