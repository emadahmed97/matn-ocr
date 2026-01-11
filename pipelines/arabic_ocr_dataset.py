#!/usr/bin/env python3
"""
Arabic OCR Dataset Pipeline for Classical Islamic Manuscripts.

This module handles loading and preprocessing of manuscript images
and their corresponding Arabic text transcriptions.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Import our Arabic text processor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from arabic_text_processing import ArabicTextProcessor


class ArabicManuscriptDataset(Dataset):
    """Dataset class for Arabic manuscript images and transcriptions."""

    def __init__(self,
                 data_dir: str,
                 image_processor=None,
                 text_processor: ArabicTextProcessor = None,
                 max_length: int = 4096,
                 image_size: Tuple[int, int] = (896, 672),
                 split: str = "train"):
        """
        Initialize the Arabic manuscript dataset.

        Args:
            data_dir: Path to dataset directory
            image_processor: Hugging Face image processor (optional)
            text_processor: Arabic text processor
            max_length: Maximum text sequence length
            image_size: Target image size (width, height)
            split: Dataset split ("train", "val", "test")
        """
        self.data_dir = Path(data_dir)
        self.image_processor = image_processor
        self.text_processor = text_processor or ArabicTextProcessor()
        self.max_length = max_length
        self.image_size = image_size
        self.split = split

        # Set up paths
        self.images_dir = self.data_dir / "images"
        self.transcriptions_dir = self.data_dir / "transcriptions"

        # Load dataset
        self.samples = self._load_dataset()

        logging.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load image-text pairs from the dataset directory."""
        samples = []

        # Method 1: Try CSV file first
        csv_file = self.data_dir / "dataset.csv"
        if csv_file.exists():
            samples = self._load_from_csv(csv_file)

        # Method 2: Try JSON metadata file
        elif (self.data_dir / "metadata.json").exists():
            samples = self._load_from_json(self.data_dir / "metadata.json")

        # Method 3: Match images with text files
        else:
            samples = self._load_from_paired_files()

        # Filter by split if specified
        if self.split != "all":
            samples = [s for s in samples if s.get("split", "train") == self.split]

        return samples

    def _load_from_csv(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load dataset from CSV file."""
        samples = []

        df = pd.read_csv(csv_file)
        required_columns = ["image_path", "text"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        for _, row in df.iterrows():
            image_path = self.data_dir / row["image_path"]
            if image_path.exists():
                sample = {
                    "image_path": image_path,
                    "text": row["text"],
                    "split": row.get("split", "train"),
                    "metadata": {
                        "source": row.get("source", "unknown"),
                        "page_number": row.get("page_number", None),
                        "book_title": row.get("book_title", None)
                    }
                }
                samples.append(sample)

        return samples

    def _load_from_json(self, json_file: Path) -> List[Dict[str, Any]]:
        """Load dataset from JSON metadata file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in data:
            image_path = self.data_dir / item["image_path"]
            if image_path.exists():
                samples.append({
                    "image_path": image_path,
                    "text": item["text"],
                    "split": item.get("split", "train"),
                    "metadata": item.get("metadata", {})
                })

        return samples

    def _load_from_paired_files(self) -> List[Dict[str, Any]]:
        """Load dataset by matching image files with text files."""
        samples = []

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        image_files = [
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        for image_file in image_files:
            # Look for corresponding text file
            text_file = self.transcriptions_dir / f"{image_file.stem}.txt"

            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                sample = {
                    "image_path": image_file,
                    "text": text,
                    "split": "train",  # Default split
                    "metadata": {
                        "filename": image_file.name,
                        "source": "manuscript"
                    }
                }
                samples.append(sample)
            else:
                logging.warning(f"No transcription found for {image_file.name}")

        return samples

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        sample = self.samples[idx]

        # Load and process image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Resize image if needed
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        # Process image with processor if available
        if self.image_processor:
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        else:
            # Convert to tensor manually
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image)

        # Process text
        processed_text = self.text_processor.preprocess_for_ocr(
            sample["text"],
            preserve_diacritics=True
        )

        return {
            "pixel_values": pixel_values,
            "text": processed_text,
            "raw_text": sample["text"],
            "image_path": str(sample["image_path"]),
            "metadata": sample["metadata"]
        }

    def create_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create train/val/test splits and save metadata."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        import random
        random.shuffle(self.samples)

        n_samples = len(self.samples)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Assign splits
        for i, sample in enumerate(self.samples):
            if i < train_end:
                sample["split"] = "train"
            elif i < val_end:
                sample["split"] = "val"
            else:
                sample["split"] = "test"

        # Save metadata with splits
        metadata_file = self.data_dir / "metadata_with_splits.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.samples, f, ensure_ascii=False, indent=2, default=str)

        logging.info(f"Created splits and saved metadata to {metadata_file}")
        logging.info(f"Train: {train_end}, Val: {val_end - train_end}, Test: {n_samples - val_end}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.samples:
            return {}

        text_lengths = []
        diacritic_ratios = []
        image_sizes = []

        for sample in self.samples:
            text = sample["text"]
            text_lengths.append(len(text))

            # Calculate diacritic ratio
            stats = self.text_processor.extract_text_statistics(text)
            diacritic_ratios.append(stats["diacritics_ratio"])

            # Get image size
            try:
                with Image.open(sample["image_path"]) as img:
                    image_sizes.append(img.size)
            except:
                pass

        stats = {
            "num_samples": len(self.samples),
            "text_length": {
                "mean": sum(text_lengths) / len(text_lengths),
                "min": min(text_lengths),
                "max": max(text_lengths),
            },
            "diacritic_ratio": {
                "mean": sum(diacritic_ratios) / len(diacritic_ratios),
                "min": min(diacritic_ratios),
                "max": max(diacritic_ratios),
            }
        }

        if image_sizes:
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            stats["image_size"] = {
                "width": {"mean": sum(widths) / len(widths), "min": min(widths), "max": max(widths)},
                "height": {"mean": sum(heights) / len(heights), "min": min(heights), "max": max(heights)},
            }

        return stats


def create_dataloaders(data_dir: str,
                      batch_size: int = 4,
                      num_workers: int = 2,
                      image_processor=None,
                      text_processor: ArabicTextProcessor = None) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_processor: Hugging Face image processor
        text_processor: Arabic text processor

    Returns:
        Dictionary with train, val, test dataloaders
    """
    dataloaders = {}

    for split in ["train", "val", "test"]:
        dataset = ArabicManuscriptDataset(
            data_dir=data_dir,
            image_processor=image_processor,
            text_processor=text_processor,
            split=split
        )

        if len(dataset) > 0:
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=collate_fn
            )

    return dataloaders


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching samples."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "texts": [item["text"] for item in batch],
        "raw_texts": [item["raw_text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "metadata": [item["metadata"] for item in batch]
    }


def setup_dataset_directory(base_dir: str) -> Path:
    """
    Set up the dataset directory structure.

    Args:
        base_dir: Base directory for the dataset

    Returns:
        Path to the created dataset directory
    """
    data_dir = Path(base_dir)

    # Create directories
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "transcriptions").mkdir(parents=True, exist_ok=True)

    # Create example CSV template
    csv_template = data_dir / "dataset_template.csv"
    if not csv_template.exists():
        with open(csv_template, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "text", "split", "source", "page_number", "book_title"])
            writer.writerow([
                "images/example.jpg",
                "بسم الله الرحمن الرحيم",
                "train",
                "manuscript",
                "1",
                "كتاب الطهارة"
            ])

    # Create README
    readme_file = data_dir / "README.md"
    if not readme_file.exists():
        readme_content = """# Arabic Manuscript Dataset

## Directory Structure
```
data/manuscripts/
├── images/           # Place your manuscript images here
├── transcriptions/   # Place corresponding text files here
├── dataset.csv       # CSV with image paths and texts (optional)
└── metadata.json     # JSON metadata file (optional)
```

## Supported Formats

### Images
- JPG, PNG, TIFF formats
- 300+ DPI recommended
- Sequential naming (e.g., page_001.jpg)

### Text Files
- UTF-8 encoded .txt files
- Same name as image (e.g., page_001.txt)
- One transcription per file

### CSV Format
```csv
image_path,text,split,source,page_number,book_title
images/page_001.jpg,"بسم الله الرحمن الرحيم",train,manuscript,1,"كتاب الطهارة"
```
"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    logging.info(f"Dataset directory structure created at: {data_dir}")
    return data_dir


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Example usage
    data_dir = setup_dataset_directory("data/manuscripts")
    print(f"Dataset directory set up at: {data_dir}")
    print(f"Please place your manuscript images in: {data_dir}/images/")
    print(f"Please place corresponding text files in: {data_dir}/transcriptions/")