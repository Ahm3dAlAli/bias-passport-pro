"""
Image Preprocessor with Bounding Box Masking

Provides image preprocessing capabilities for bias analysis:
- Face detection and bounding box extraction
- Selective masking (face, body, background)
- Counterfactual image generation
- Image normalization and resizing

Key use cases:
1. Mask faces to test if model bias persists without facial features
2. Mask backgrounds to isolate person from context
3. Generate counterfactual pairs by swapping backgrounds
4. Standardize image inputs across experiments
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class MaskingStrategy(Enum):
    """Masking strategies for counterfactual experiments."""

    NONE = "none"                    # No masking
    FACE_BLUR = "face_blur"          # Blur face region
    FACE_BLACK = "face_black"        # Black out face
    FACE_WHITE = "face_white"        # White out face
    FACE_NOISE = "face_noise"        # Replace face with noise
    BACKGROUND_BLUR = "background_blur"  # Blur everything except person
    BACKGROUND_GRAY = "background_gray"  # Gray background
    SILHOUETTE = "silhouette"        # Show only person silhouette


@dataclass
class BoundingBox:
    """Bounding box for detected regions."""

    x: int          # Top-left x coordinate
    y: int          # Top-left y coordinate
    width: int      # Box width
    height: int     # Box height
    confidence: float = 1.0  # Detection confidence

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of the box."""
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) tuple."""
        return (self.x, self.y, self.x2, self.y2)

    def expand(self, factor: float = 1.2) -> "BoundingBox":
        """Expand box by a factor."""
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)
        new_x = self.x - (new_width - self.width) // 2
        new_y = self.y - (new_height - self.height) // 2

        return BoundingBox(
            x=max(0, new_x),
            y=max(0, new_y),
            width=new_width,
            height=new_height,
            confidence=self.confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class FaceDetection:
    """Result of face detection."""

    face_box: BoundingBox
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None  # eye_left, eye_right, nose, mouth_left, mouth_right
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_box": self.face_box.to_dict(),
            "landmarks": self.landmarks,
            "attributes": self.attributes,
        }


class ImagePreprocessor:
    """
    Image preprocessor for bias analysis experiments.

    Provides:
    - Face detection (using multiple backends)
    - Bounding box masking
    - Image normalization
    - Counterfactual generation

    Example:
        >>> processor = ImagePreprocessor()
        >>> # Detect face
        >>> detections = processor.detect_faces(image)
        >>> # Mask face to test if bias persists
        >>> masked = processor.apply_mask(image, detections[0].face_box, MaskingStrategy.FACE_BLUR)
        >>> # Compare model responses on original vs masked
    """

    def __init__(
        self,
        detector_backend: str = "opencv",  # opencv, mtcnn, mediapipe
        target_size: Optional[Tuple[int, int]] = None,  # (width, height)
        normalize: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            detector_backend: Face detection backend to use
            target_size: Optional target size for resizing
            normalize: Whether to normalize pixel values
        """
        self.detector_backend = detector_backend
        self.target_size = target_size
        self.normalize = normalize

        self._detector = None

    def _get_detector(self):
        """Lazy-load face detector."""
        if self._detector is not None:
            return self._detector

        if self.detector_backend == "opencv":
            self._detector = self._create_opencv_detector()
        elif self.detector_backend == "mtcnn":
            self._detector = self._create_mtcnn_detector()
        elif self.detector_backend == "mediapipe":
            self._detector = self._create_mediapipe_detector()
        else:
            # Fallback to simple OpenCV
            self._detector = self._create_opencv_detector()

        return self._detector

    def _create_opencv_detector(self):
        """Create OpenCV cascade detector."""
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        except Exception:
            return None

    def _create_mtcnn_detector(self):
        """Create MTCNN detector."""
        try:
            from mtcnn import MTCNN
            return MTCNN()
        except ImportError:
            print("MTCNN not installed. Falling back to OpenCV.")
            return self._create_opencv_detector()

    def _create_mediapipe_detector(self):
        """Create MediaPipe detector."""
        try:
            import mediapipe as mp
            return mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )
        except ImportError:
            print("MediaPipe not installed. Falling back to OpenCV.")
            return self._create_opencv_detector()

    def detect_faces(
        self,
        image: Union[str, Path, Image.Image],
        min_confidence: float = 0.5,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image: Image path or PIL Image
            min_confidence: Minimum detection confidence

        Returns:
            List of FaceDetection objects
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        img_array = np.array(image)
        detections = []

        detector = self._get_detector()
        if detector is None:
            return detections

        if self.detector_backend == "opencv":
            detections = self._detect_opencv(img_array, detector, min_confidence)
        elif self.detector_backend == "mtcnn":
            detections = self._detect_mtcnn(img_array, detector, min_confidence)
        elif self.detector_backend == "mediapipe":
            detections = self._detect_mediapipe(img_array, image, detector, min_confidence)

        return detections

    def _detect_opencv(
        self,
        img_array: np.ndarray,
        detector,
        min_confidence: float,
    ) -> List[FaceDetection]:
        """Detect faces using OpenCV."""
        import cv2

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        detections = []
        for (x, y, w, h) in faces:
            detections.append(FaceDetection(
                face_box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
            ))

        return detections

    def _detect_mtcnn(
        self,
        img_array: np.ndarray,
        detector,
        min_confidence: float,
    ) -> List[FaceDetection]:
        """Detect faces using MTCNN."""
        results = detector.detect_faces(img_array)

        detections = []
        for result in results:
            if result['confidence'] >= min_confidence:
                box = result['box']
                keypoints = result['keypoints']

                landmarks = {
                    'eye_left': keypoints.get('left_eye'),
                    'eye_right': keypoints.get('right_eye'),
                    'nose': keypoints.get('nose'),
                    'mouth_left': keypoints.get('mouth_left'),
                    'mouth_right': keypoints.get('mouth_right'),
                }

                detections.append(FaceDetection(
                    face_box=BoundingBox(
                        x=box[0], y=box[1],
                        width=box[2], height=box[3],
                        confidence=result['confidence'],
                    ),
                    landmarks=landmarks,
                ))

        return detections

    def _detect_mediapipe(
        self,
        img_array: np.ndarray,
        pil_image: Image.Image,
        detector,
        min_confidence: float,
    ) -> List[FaceDetection]:
        """Detect faces using MediaPipe."""
        results = detector.process(img_array)
        detections = []

        if results.detections:
            img_h, img_w = img_array.shape[:2]

            for detection in results.detections:
                if detection.score[0] >= min_confidence:
                    bbox = detection.location_data.relative_bounding_box

                    x = int(bbox.xmin * img_w)
                    y = int(bbox.ymin * img_h)
                    w = int(bbox.width * img_w)
                    h = int(bbox.height * img_h)

                    detections.append(FaceDetection(
                        face_box=BoundingBox(
                            x=max(0, x), y=max(0, y),
                            width=w, height=h,
                            confidence=detection.score[0],
                        ),
                    ))

        return detections

    def apply_mask(
        self,
        image: Union[str, Path, Image.Image],
        box: BoundingBox,
        strategy: MaskingStrategy = MaskingStrategy.FACE_BLUR,
        expand_factor: float = 1.0,
    ) -> Image.Image:
        """
        Apply masking to a region of the image.

        Args:
            image: Input image
            box: Bounding box to mask
            strategy: Masking strategy to use
            expand_factor: Factor to expand the box before masking

        Returns:
            Masked PIL Image
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.copy()

        # Expand box if requested
        if expand_factor != 1.0:
            box = box.expand(expand_factor)

        # Clamp box to image boundaries
        box = BoundingBox(
            x=max(0, box.x),
            y=max(0, box.y),
            width=min(box.width, image.width - box.x),
            height=min(box.height, image.height - box.y),
            confidence=box.confidence,
        )

        if strategy == MaskingStrategy.NONE:
            return image

        elif strategy == MaskingStrategy.FACE_BLUR:
            return self._apply_blur_mask(image, box, blur_radius=30)

        elif strategy == MaskingStrategy.FACE_BLACK:
            return self._apply_solid_mask(image, box, color=(0, 0, 0))

        elif strategy == MaskingStrategy.FACE_WHITE:
            return self._apply_solid_mask(image, box, color=(255, 255, 255))

        elif strategy == MaskingStrategy.FACE_NOISE:
            return self._apply_noise_mask(image, box)

        elif strategy == MaskingStrategy.BACKGROUND_BLUR:
            return self._apply_inverse_blur(image, box)

        elif strategy == MaskingStrategy.BACKGROUND_GRAY:
            return self._apply_inverse_gray(image, box)

        elif strategy == MaskingStrategy.SILHOUETTE:
            return self._apply_silhouette(image, box)

        return image

    def _apply_blur_mask(
        self,
        image: Image.Image,
        box: BoundingBox,
        blur_radius: int = 30,
    ) -> Image.Image:
        """Apply blur to region."""
        region = image.crop(box.to_tuple())
        blurred = region.filter(ImageFilter.GaussianBlur(blur_radius))
        image.paste(blurred, (box.x, box.y))
        return image

    def _apply_solid_mask(
        self,
        image: Image.Image,
        box: BoundingBox,
        color: Tuple[int, int, int],
    ) -> Image.Image:
        """Apply solid color to region."""
        draw = ImageDraw.Draw(image)
        draw.rectangle(box.to_tuple(), fill=color)
        return image

    def _apply_noise_mask(
        self,
        image: Image.Image,
        box: BoundingBox,
    ) -> Image.Image:
        """Apply random noise to region."""
        noise = np.random.randint(0, 256, (box.height, box.width, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise)
        image.paste(noise_img, (box.x, box.y))
        return image

    def _apply_inverse_blur(
        self,
        image: Image.Image,
        box: BoundingBox,
    ) -> Image.Image:
        """Blur everything except the region."""
        # Blur entire image
        blurred = image.filter(ImageFilter.GaussianBlur(20))

        # Paste original region back
        region = image.crop(box.to_tuple())
        blurred.paste(region, (box.x, box.y))

        return blurred

    def _apply_inverse_gray(
        self,
        image: Image.Image,
        box: BoundingBox,
    ) -> Image.Image:
        """Gray background, keep person in color."""
        # Convert to grayscale
        gray = image.convert("L").convert("RGB")

        # Paste original region back
        region = image.crop(box.to_tuple())
        gray.paste(region, (box.x, box.y))

        return gray

    def _apply_silhouette(
        self,
        image: Image.Image,
        box: BoundingBox,
    ) -> Image.Image:
        """Create silhouette effect."""
        # Create white background
        result = Image.new("RGB", image.size, (255, 255, 255))

        # Create black silhouette of face region
        draw = ImageDraw.Draw(result)
        draw.ellipse(box.to_tuple(), fill=(0, 0, 0))

        return result

    def preprocess(
        self,
        image: Union[str, Path, Image.Image],
        detect_faces: bool = True,
        apply_masking: Optional[MaskingStrategy] = None,
    ) -> Tuple[Image.Image, List[FaceDetection]]:
        """
        Full preprocessing pipeline.

        Args:
            image: Input image
            detect_faces: Whether to detect faces
            apply_masking: Optional masking strategy

        Returns:
            Tuple of (processed image, face detections)
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        detections = []

        # Resize if target size specified
        if self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Detect faces
        if detect_faces:
            detections = self.detect_faces(image)

        # Apply masking
        if apply_masking and detections:
            for detection in detections:
                image = self.apply_mask(image, detection.face_box, apply_masking)

        return image, detections

    def create_counterfactual_pair(
        self,
        image: Union[str, Path, Image.Image],
        masking_strategy: MaskingStrategy = MaskingStrategy.FACE_BLUR,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create a counterfactual pair: original + masked version.

        Useful for testing if model bias persists without facial features.

        Args:
            image: Input image
            masking_strategy: How to mask the face

        Returns:
            Tuple of (original, masked) images
        """
        # Load image
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert("RGB")
        else:
            original = image.convert("RGB")

        # Detect faces
        detections = self.detect_faces(original)

        if not detections:
            # No face detected, return original twice
            return original, original.copy()

        # Apply masking to copy
        masked = original.copy()
        for detection in detections:
            masked = self.apply_mask(
                masked,
                detection.face_box.expand(1.2),  # Slightly expand to ensure full coverage
                masking_strategy,
            )

        return original, masked

    def batch_preprocess(
        self,
        images: List[Union[str, Path, Image.Image]],
        detect_faces: bool = True,
        apply_masking: Optional[MaskingStrategy] = None,
    ) -> List[Tuple[Image.Image, List[FaceDetection]]]:
        """
        Preprocess a batch of images.

        Args:
            images: List of images
            detect_faces: Whether to detect faces
            apply_masking: Optional masking strategy

        Returns:
            List of (processed image, detections) tuples
        """
        results = []
        for image in images:
            result = self.preprocess(image, detect_faces, apply_masking)
            results.append(result)
        return results
