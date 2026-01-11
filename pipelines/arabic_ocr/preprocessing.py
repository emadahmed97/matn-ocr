"""
Image preprocessing utilities for Arabic OCR.

Handles image transformations, augmentations, and format conversions
for Arabic manuscript processing.
"""

import io
from typing import Tuple, Union, Optional
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np


class ArabicImageProcessor:
    """
    Image processor specialized for Arabic OCR tasks.

    Handles various image formats and provides preprocessing
    suitable for classical Arabic manuscripts.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        background_color: Tuple[int, int, int] = (255, 255, 255),
        normalize: bool = True
    ):
        """
        Initialize the Arabic image processor.

        Args:
            target_size: Target size for processed images
            background_color: Background color for padding
            normalize: Whether to normalize pixel values
        """
        self.target_size = target_size
        self.background_color = background_color
        self.normalize = normalize

    def load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Load image from various input formats.

        Args:
            image_input: Image path, bytes, or PIL Image

        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image_input, str):
            # File path
            image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            # Bytes
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Convert to RGB (handles grayscale, RGBA, etc.)
        return image.convert("RGB")

    def enhance_text_image(self, image: Image.Image, enhance_contrast: bool = True) -> Image.Image:
        """
        Enhance image quality for better OCR performance.

        Args:
            image: Input PIL Image
            enhance_contrast: Whether to enhance contrast

        Returns:
            Enhanced PIL Image
        """
        # Convert to grayscale for processing
        gray_image = image.convert("L")

        # Enhance contrast if requested
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(gray_image)
            gray_image = enhancer.enhance(1.2)

        # Apply slight sharpening
        gray_image = gray_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # Convert back to RGB
        return gray_image.convert("RGB")

    def pad_to_aspect_ratio(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Pad image to target aspect ratio while maintaining content.

        Args:
            image: Input PIL Image
            target_size: Target size (width, height)

        Returns:
            Padded PIL Image
        """
        if target_size is None:
            target_size = self.target_size

        # Calculate padding to maintain aspect ratio
        target_ratio = target_size[0] / target_size[1]
        current_ratio = image.width / image.height

        if current_ratio > target_ratio:
            # Image is wider - pad height
            new_width = target_size[0]
            new_height = int(target_size[0] / current_ratio)
        else:
            # Image is taller - pad width
            new_width = int(target_size[1] * current_ratio)
            new_height = target_size[1]

        # Resize image
        resized = image.resize((new_width, new_height), Image.LANCZOS)

        # Pad to exact target size
        padded = ImageOps.pad(
            resized,
            target_size,
            color=self.background_color,
            centering=(0.5, 0.5)
        )

        return padded

    def resize_with_padding(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image with optional padding to maintain aspect ratio.

        Args:
            image: Input PIL Image
            target_size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio with padding

        Returns:
            Resized PIL Image
        """
        if target_size is None:
            target_size = self.target_size

        if maintain_aspect:
            return self.pad_to_aspect_ratio(image, target_size)
        else:
            return image.resize(target_size, Image.LANCZOS)

    def augment_image(self, image: Image.Image, augmentation_level: str = "light") -> Image.Image:
        """
        Apply data augmentation suitable for OCR training.

        Args:
            image: Input PIL Image
            augmentation_level: "light", "medium", or "heavy"

        Returns:
            Augmented PIL Image
        """
        augmented = image.copy()

        if augmentation_level == "light":
            # Light augmentation - suitable for real manuscripts
            # Slight rotation
            angle = np.random.uniform(-2, 2)
            augmented = augmented.rotate(angle, expand=True, fillcolor=self.background_color)

            # Slight brightness adjustment
            enhancer = ImageEnhance.Brightness(augmented)
            brightness_factor = np.random.uniform(0.9, 1.1)
            augmented = enhancer.enhance(brightness_factor)

        elif augmentation_level == "medium":
            # Medium augmentation
            # Rotation
            angle = np.random.uniform(-5, 5)
            augmented = augmented.rotate(angle, expand=True, fillcolor=self.background_color)

            # Brightness and contrast
            brightness_factor = np.random.uniform(0.8, 1.2)
            contrast_factor = np.random.uniform(0.9, 1.1)

            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(brightness_factor)

            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(contrast_factor)

        elif augmentation_level == "heavy":
            # Heavy augmentation - use sparingly for Arabic text
            # Rotation
            angle = np.random.uniform(-10, 10)
            augmented = augmented.rotate(angle, expand=True, fillcolor=self.background_color)

            # Multiple enhancements
            brightness_factor = np.random.uniform(0.7, 1.3)
            contrast_factor = np.random.uniform(0.8, 1.2)
            sharpness_factor = np.random.uniform(0.9, 1.1)

            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(brightness_factor)

            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(contrast_factor)

            enhancer = ImageEnhance.Sharpness(augmented)
            augmented = enhancer.enhance(sharpness_factor)

            # Add slight blur occasionally
            if np.random.random() < 0.3:
                augmented = augmented.filter(ImageFilter.GaussianBlur(radius=0.5))

        return augmented

    def preprocess_for_ocr(
        self,
        image_input: Union[str, bytes, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        enhance: bool = True,
        augment: Optional[str] = None
    ) -> Image.Image:
        """
        Complete preprocessing pipeline for OCR.

        Args:
            image_input: Input image (path, bytes, or PIL Image)
            target_size: Target size for output
            enhance: Whether to enhance image quality
            augment: Augmentation level ("light", "medium", "heavy", or None)

        Returns:
            Preprocessed PIL Image ready for OCR
        """
        # Load image
        image = self.load_image(image_input)

        # Enhance if requested
        if enhance:
            image = self.enhance_text_image(image)

        # Apply augmentation if specified
        if augment:
            image = self.augment_image(image, augment)

        # Resize with padding
        image = self.resize_with_padding(image, target_size)

        return image

    def batch_preprocess(
        self,
        image_inputs: list,
        target_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> list:
        """
        Preprocess a batch of images.

        Args:
            image_inputs: List of image inputs
            target_size: Target size for output
            **kwargs: Additional arguments for preprocess_for_ocr

        Returns:
            List of preprocessed PIL Images
        """
        return [
            self.preprocess_for_ocr(img_input, target_size, **kwargs)
            for img_input in image_inputs
        ]


def create_synthetic_manuscript_style(
    text: str,
    font_size: int = 24,
    image_size: Tuple[int, int] = (800, 600),
    background_color: Tuple[int, int, int] = (245, 245, 240),
    text_color: Tuple[int, int, int] = (20, 20, 20)
) -> Image.Image:
    """
    Create a synthetic manuscript-style image from Arabic text.

    This is useful for data augmentation when you have text but need images.

    Args:
        text: Arabic text to render
        font_size: Size of the font
        image_size: Size of the output image
        background_color: Background color
        text_color: Text color

    Returns:
        PIL Image with rendered Arabic text
    """
    # Create a blank image
    image = Image.new("RGB", image_size, background_color)

    # Note: For proper Arabic text rendering, you would need:
    # 1. Arabic font files (like Amiri, Scheherazade, etc.)
    # 2. Proper RTL text handling
    # 3. Shaping engine for connected letters

    # This is a placeholder implementation
    # In production, use libraries like PIL with Arabic fonts,
    # or integrate with specialized Arabic text rendering libraries

    return image


# Utility functions for image format conversions
def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    byte_io = io.BytesIO()
    image.save(byte_io, format=format)
    return byte_io.getvalue()


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes))


def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)


def pil_to_numpy(image: Image.Image, normalize: bool = False) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    array = np.array(image)
    if normalize:
        array = array.astype(np.float32) / 255.0
    return array