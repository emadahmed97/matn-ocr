"""
Arabic OCR module for classical Islamic texts.

This module provides components for training and deploying Arabic OCR models
using DeepSeek-OCR with Unsloth fine-tuning, adapted from the ML School framework.
"""

from .model import load_deepseek_ocr_model
from .data_collator import DeepSeekOCRDataCollator
from .preprocessing import ArabicImageProcessor
from .metrics import calculate_cer, calculate_wer, calculate_bleu

__all__ = [
    "load_deepseek_ocr_model",
    "DeepSeekOCRDataCollator",
    "ArabicImageProcessor",
    "calculate_cer",
    "calculate_wer",
    "calculate_bleu",
]