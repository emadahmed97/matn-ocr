#!/usr/bin/env python3
"""
Arabic text processing pipeline for Classical Islamic texts.

This module provides utilities for normalizing, cleaning, and preprocessing
Arabic text specifically for OCR training on classical Islamic manuscripts.
"""

import re
import unicodedata
from typing import List, Dict, Tuple


class ArabicTextProcessor:
    """Handles Arabic text normalization and preprocessing for OCR."""

    def __init__(self):
        """Initialize the Arabic text processor with normalization rules."""
        # Arabic Unicode ranges
        self.arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic block
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]

        # Common Arabic diacritics
        self.diacritics = [
            '\u064B',  # Fathatan
            '\u064C',  # Dammatan
            '\u064D',  # Kasratan
            '\u064E',  # Fatha
            '\u064F',  # Damma
            '\u0650',  # Kasra
            '\u0651',  # Shadda
            '\u0652',  # Sukun
            '\u0653',  # Maddah
            '\u0654',  # Hamza above
            '\u0655',  # Hamza below
            '\u0656',  # Subscript alef
            '\u0657',  # Inverted damma
            '\u0658',  # Mark noon ghunna
        ]

        # Islamic text markers and abbreviations (disabled for now to prevent text corruption)
        self.islamic_abbreviations = {
            # 'ص': 'صلى الله عليه وسلم',  # PBUH
            # 'رض': 'رضي الله عنه',      # May Allah be pleased with him
            # 'رحمه': 'رحمه الله',        # May Allah have mercy on him
            # 'ع': 'عليه السلام',        # Peace be upon him
        }

        # Classical Arabic normalization rules
        self.normalization_rules = [
            # Alef variations
            ('\u0622', '\u0627'),  # Alef with madda -> alef
            ('\u0623', '\u0627'),  # Alef with hamza above -> alef
            ('\u0625', '\u0627'),  # Alef with hamza below -> alef
            ('\u0671', '\u0627'),  # Alef wasla -> alef

            # Yeh variations
            ('\u064A', '\u064A'),  # Yeh (normalize to standard)
            ('\u0649', '\u064A'),  # Alef maksura -> yeh

            # Heh variations
            ('\u0629', '\u0647'),  # Teh marbuta -> heh

            # Remove tatweel (kashida)
            ('\u0640', ''),        # Tatweel
        ]

    def is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        for char in text:
            code_point = ord(char)
            for start, end in self.arabic_ranges:
                if start <= code_point <= end:
                    return True
        return False

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics from text."""
        for diacritic in self.diacritics:
            text = text.replace(diacritic, '')
        return text

    def preserve_diacritics(self, text: str) -> str:
        """Preserve and normalize diacritics in text."""
        # Normalize common diacritic combinations
        text = re.sub(r'[\u064B-\u0652]{2,}', lambda m: m.group(0)[-1], text)
        return text

    def normalize_arabic(self, text: str) -> str:
        """Apply normalization rules to Arabic text."""
        for old_char, new_char in self.normalization_rules:
            text = text.replace(old_char, new_char)
        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand common Islamic abbreviations."""
        for abbrev, expansion in self.islamic_abbreviations.items():
            text = text.replace(abbrev, expansion)
        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for OCR training."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove non-printable characters except Arabic
        text = ''.join(char for char in text if
                      unicodedata.category(char)[0] != 'C' or
                      self.is_arabic_text(char))

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        return text.strip()

    def preprocess_for_ocr(self, text: str, preserve_diacritics: bool = True) -> str:
        """
        Complete preprocessing pipeline for OCR training.

        Args:
            text: Input Arabic text
            preserve_diacritics: Whether to keep diacritics

        Returns:
            Preprocessed text ready for OCR training
        """
        # Basic cleaning
        text = self.clean_text(text)

        # Normalize Arabic characters
        text = self.normalize_arabic(text)

        # Handle diacritics
        if preserve_diacritics:
            text = self.preserve_diacritics(text)
        else:
            text = self.remove_diacritics(text)

        # Expand abbreviations for better training
        text = self.expand_abbreviations(text)

        return text

    def extract_text_statistics(self, text: str) -> Dict[str, int]:
        """Extract statistics about Arabic text."""
        stats = {
            'total_chars': len(text),
            'arabic_chars': 0,
            'diacritics_count': 0,
            'words': len(text.split()),
            'lines': len(text.split('\n')),
        }

        for char in text:
            if self.is_arabic_text(char):
                stats['arabic_chars'] += 1
            if char in self.diacritics:
                stats['diacritics_count'] += 1

        stats['diacritics_ratio'] = (
            stats['diacritics_count'] / stats['arabic_chars']
            if stats['arabic_chars'] > 0 else 0
        )

        return stats


def test_arabic_processing():
    """Test the Arabic text processing pipeline."""
    output = []
    output.append("# 🔤 Arabic Text Processing Pipeline Test Results\n")

    processor = ArabicTextProcessor()

    # Test texts
    test_texts = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",  # With diacritics
        "الحمد لله رب العالمين",                    # Without diacritics
        "قال رسول الله ص: اطلبوا العلم",            # With abbreviation
        "كتاب الطهارة - باب الوضوء",               # Classical text format
        "وَقَالَ الْإِمَامُ أَحْمَدُ رحمه اللَّهُ",    # Mixed diacritics
    ]

    for i, text in enumerate(test_texts, 1):
        output.append(f"\n## Test {i}\n")
        output.append(f"**Original Text:**\n```\n{text}\n```\n")

        # Test different processing options
        clean_with_diacritics = processor.preprocess_for_ocr(text, preserve_diacritics=True)
        clean_without_diacritics = processor.preprocess_for_ocr(text, preserve_diacritics=False)

        output.append(f"**With Diacritics:**\n```\n{clean_with_diacritics}\n```\n")
        output.append(f"**Without Diacritics:**\n```\n{clean_without_diacritics}\n```\n")

        # Get statistics
        stats = processor.extract_text_statistics(text)
        output.append(f"**Statistics:**\n")
        for key, value in stats.items():
            output.append(f"- {key}: {value}\n")

    output.append("\n✅ **Arabic text processing pipeline test completed!**\n")
    return "\n".join(output)


def analyze_dataset_sample():
    """Analyze a sample from the Arabic books dataset."""
    output = []
    output.append("\n# 📊 Dataset Sample Analysis\n")

    processor = ArabicTextProcessor()

    # Sample Islamic text (classical style)
    sample_text = """
    كتاب الطهارة

    باب الوضوء وفرائضه وسننه

    قال الله تعالى: يا أيها الذين آمنوا إذا قمتم إلى الصلاة فاغسلوا وجوهكم وأيديكم إلى المرافق وامسحوا برؤوسكم وأرجلكم إلى الكعبين

    وعن أبي هريرة رض قال: قال رسول الله ص: لا يقبل الله صلاة أحدكم إذا أحدث حتى يتوضأ

    فصل في آداب الوضوء:
    ١. البسملة عند البدء
    ٢. غسل الكفين ثلاثاً
    ٣. المضمضة والاستنشاق
    """

    output.append("## Sample Classical Islamic Text\n")
    output.append("```arabic\n")
    output.append(sample_text.strip())
    output.append("\n```\n")

    # Analyze the sample
    stats = processor.extract_text_statistics(sample_text)
    output.append("## Text Statistics\n")
    for key, value in stats.items():
        output.append(f"- **{key}**: {value}\n")

    # Test preprocessing
    processed = processor.preprocess_for_ocr(sample_text)
    output.append("\n## Processed Text Preview\n")
    output.append("```arabic\n")
    output.append(processed[:400] + "..." if len(processed) > 400 else processed)
    output.append("\n```\n")

    # Check for Islamic patterns
    islamic_terms = ["الله", "رسول", "قال", "باب", "كتاب", "فصل"]
    found_terms = [term for term in islamic_terms if term in sample_text]
    output.append("## Islamic Text Pattern Analysis\n")
    output.append(f"- **Contains Islamic terminology**: {len(found_terms) > 0}\n")
    output.append(f"- **Found terms**: {', '.join(found_terms)}\n")

    return "\n".join(output)


def main():
    """Main function to test Arabic processing capabilities."""
    print("🕌 Generating Arabic Text Processing Analysis Report...")

    # Generate complete report
    report = []
    report.append("# 🕌 Arabic Text Processing for Classical Islamic Texts\n")
    report.append("*Generated by ML School Arabic OCR Pipeline*\n")

    # Run tests and collect output
    report.append(test_arabic_processing())
    report.append(analyze_dataset_sample())

    report.append("\n# 🎯 Next Steps\n")
    report.append("1. **Apply this processing to the full Arabic books dataset**\n")
    report.append("2. **Create image-text pairs for OCR training**\n")
    report.append("3. **Implement evaluation metrics for Arabic OCR**\n")
    report.append("4. **Fine-tune Nougat model on Arabic manuscripts**\n")

    # Write to markdown file
    output_file = "arabic_text_analysis_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✅ Report generated: {output_file}")
    print(f"📄 View the report for detailed Arabic text analysis results")


if __name__ == "__main__":
    main()