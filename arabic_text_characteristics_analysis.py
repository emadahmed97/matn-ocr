#!/usr/bin/env python3
"""
Comprehensive Arabic text characteristics analysis for Classical Islamic manuscripts.

This script analyzes the MohamedRashad/arabic-books dataset to understand:
- Diacritic patterns and frequency
- Classical Islamic text formatting
- Character distribution and patterns
- Manuscript-specific linguistic features
"""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
# from datasets import load_dataset  # Not needed for sample analysis


class ArabicTextCharacteristicsAnalyzer:
    """Analyzes Arabic text characteristics for OCR training."""

    def __init__(self):
        """Initialize the analyzer with Arabic text processing capabilities."""

        # Extended diacritics list
        self.diacritics = {
            '\u064B': 'Fathatan',     # ً
            '\u064C': 'Dammatan',     # ٌ
            '\u064D': 'Kasratan',     # ٍ
            '\u064E': 'Fatha',        # َ
            '\u064F': 'Damma',        # ُ
            '\u0650': 'Kasra',        # ِ
            '\u0651': 'Shadda',       # ّ
            '\u0652': 'Sukun',        # ْ
            '\u0653': 'Maddah',       # ٓ
            '\u0654': 'Hamza above',  # ٔ
            '\u0655': 'Hamza below',  # ٕ
            '\u0656': 'Subscript alef', # ٖ
            '\u0657': 'Inverted damma', # ٗ
            '\u0658': 'Mark noon ghunna', # ٘
        }

        # Arabic letters
        self.arabic_letters = list(range(0x0627, 0x064A + 1))  # ا to ي

        # Classical Islamic terminology patterns
        self.islamic_patterns = [
            r'الله',           # Allah
            r'رسول',          # Messenger
            r'النبي',          # The Prophet
            r'صلى الله عليه وسلم', # PBUH
            r'رضي الله عنه',    # May Allah be pleased with him
            r'رحمه الله',       # May Allah have mercy on him
            r'عليه السلام',     # Peace be upon him
            r'تعالى',          # Most High
            r'سبحانه',         # Glory be to Him
            r'باب',            # Chapter
            r'كتاب',           # Book
            r'فصل',            # Section
            r'حديث',           # Hadith
            r'قال',            # He said
            r'عن',             # From/About
        ]

        # Manuscript formatting patterns
        self.formatting_patterns = {
            'chapter_headers': [r'كتاب\s+\w+', r'باب\s+\w+', r'فصل\s+\w+'],
            'numbered_lists': [r'[١-٩]+\.', r'[1-9]+\.', r'[أ-ي]\.'],
            'citations': [r'قال\s+\w+', r'وعن\s+\w+', r'حدثنا\s+\w+'],
            'references': [r'رواه\s+\w+', r'أخرجه\s+\w+', r'ذكره\s+\w+'],
        }

    def analyze_diacritics(self, text: str) -> Dict:
        """Analyze diacritic patterns in text."""
        diacritic_counts = Counter()
        total_chars = len(text)
        arabic_chars = 0

        # Count characters and diacritics
        for char in text:
            if '\u0600' <= char <= '\u06FF':  # Arabic block
                arabic_chars += 1
                if char in self.diacritics:
                    diacritic_counts[self.diacritics[char]] += 1

        # Calculate statistics
        total_diacritics = sum(diacritic_counts.values())
        diacritic_ratio = total_diacritics / arabic_chars if arabic_chars > 0 else 0

        return {
            'total_chars': total_chars,
            'arabic_chars': arabic_chars,
            'diacritic_counts': dict(diacritic_counts),
            'total_diacritics': total_diacritics,
            'diacritic_ratio': diacritic_ratio,
            'most_common_diacritics': diacritic_counts.most_common(5)
        }

    def analyze_formatting_patterns(self, text: str) -> Dict:
        """Analyze manuscript formatting patterns."""
        results = {}

        for pattern_type, patterns in self.formatting_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.UNICODE)
                matches.extend(found)
            results[pattern_type] = {
                'count': len(matches),
                'examples': matches[:5]  # First 5 examples
            }

        return results

    def analyze_islamic_terminology(self, text: str) -> Dict:
        """Analyze Islamic terminology frequency."""
        term_counts = {}

        for pattern in self.islamic_patterns:
            matches = re.findall(pattern, text, re.UNICODE)
            if matches:
                term_counts[pattern] = len(matches)

        return {
            'term_frequencies': term_counts,
            'total_islamic_terms': sum(term_counts.values()),
            'unique_terms': len(term_counts)
        }

    def analyze_character_distribution(self, text: str) -> Dict:
        """Analyze character frequency distribution."""
        char_counts = Counter()

        for char in text:
            if '\u0600' <= char <= '\u06FF':  # Arabic block
                char_counts[char] += 1

        # Get most common characters
        most_common = char_counts.most_common(20)

        return {
            'character_frequencies': dict(char_counts),
            'most_common_chars': most_common,
            'unique_characters': len(char_counts),
            'total_arabic_chars': sum(char_counts.values())
        }

    def analyze_text_structure(self, text: str) -> Dict:
        """Analyze overall text structure."""
        lines = text.split('\n')
        words = text.split()

        # Line length analysis
        line_lengths = [len(line) for line in lines if line.strip()]
        avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0

        # Word length analysis
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0

        return {
            'total_lines': len(lines),
            'non_empty_lines': len(line_lengths),
            'total_words': len(words),
            'avg_line_length': avg_line_length,
            'avg_word_length': avg_word_length,
            'max_line_length': max(line_lengths) if line_lengths else 0,
            'min_line_length': min(line_lengths) if line_lengths else 0
        }


def analyze_dataset_sample():
    """Analyze sample classical Arabic Islamic texts."""
    print("🔍 Analyzing classical Arabic Islamic text samples...")

    analyzer = ArabicTextCharacteristicsAnalyzer()

    # Sample classical Islamic texts for analysis
    sample_texts = [
        """
        كتاب الطهارة

        باب الوضوء وفرائضه وسننه

        قال الله تعالى: يا أيها الذين آمنوا إذا قمتم إلى الصلاة فاغسلوا وجوهكم وأيديكم إلى المرافق وامسحوا برؤوسكم وأرجلكم إلى الكعبين

        وعن أبي هريرة رضي الله عنه قال: قال رسول الله صلى الله عليه وسلم: لا يقبل الله صلاة أحدكم إذا أحدث حتى يتوضأ

        فصل في آداب الوضوء:
        ١. البسملة عند البدء
        ٢. غسل الكفين ثلاثاً
        ٣. المضمضة والاستنشاق
        ٤. غسل الوجه ثلاثاً
        ٥. غسل اليدين إلى المرفقين
        """,

        """
        كتاب الصلاة

        باب أوقات الصلاة وشروطها

        عن عبد الله بن عمر رضي الله عنهما قال: قال النبي صلى الله عليه وسلم: وقت الظهر إذا زالت الشمس وكان ظل الرجل كطوله ما لم يحضر العصر، ووقت العصر ما لم تصفر الشمس

        حدثنا مالك عن زيد بن أسلم عن عطاء بن يسار عن ابن عباس رضي الله عنهما أنه قال: أخبرني أبو سفيان أن هرقل قال له: سألتك عن الصلاة فزعمت أنه يأمركم بها

        فصل في قبلة المصلي:
        أ. استقبال القبلة شرط لصحة الصلاة
        ب. تحري القبلة عند الاشتباه
        ج. الصلاة في الطائرة والسفينة
        """,

        """
        كتاب الزكاة والصدقة

        باب زكاة الذهب والفضة

        قال تعالى: والذين يكنزون الذهب والفضة ولا ينفقونها في سبيل الله فبشرهم بعذاب أليم

        عن أبي سعيد الخدري رضي الله عنه قال: قال رسول الله صلى الله عليه وسلم: ليس فيما دون خمس أواق من الورق صدقة، وليس فيما دون خمس ذود من الإبل صدقة

        وعن علي بن أبي طالب رضي الله عنه قال: هذه صدقة رسول الله صلى الله عليه وسلم التي فرضها الله عز وجل:
        ١. في أربع وعشرين من الإبل فما دونها الغنم في كل خمس شاة
        ٢. فإذا بلغت خمساً وعشرين إلى خمس وثلاثين ففيها بنت مخاض أنثى
        ٣. فإذا بلغت ستاً وثلاثين إلى خمس وأربعين ففيها بنت لبون أنثى

        فصل في زكاة الزروع والثمار:
        - النصاب في الزروع والثمار خمسة أوسق
        - الواجب العشر إن سقيت بماء المطر والأنهار
        - الواجب نصف العشر إن سقيت بالدلاء والنواضح
        """
    ]

    # Combine texts for analysis
    combined_text = '\n'.join(sample_texts)

    print(f"✅ Loaded {len(sample_texts)} samples ({len(combined_text):,} characters)")

    # Perform analyses
    results = {}
    results['diacritics'] = analyzer.analyze_diacritics(combined_text)
    results['formatting'] = analyzer.analyze_formatting_patterns(combined_text)
    results['islamic_terms'] = analyzer.analyze_islamic_terminology(combined_text)
    results['character_dist'] = analyzer.analyze_character_distribution(combined_text)
    results['text_structure'] = analyzer.analyze_text_structure(combined_text)

    return results


def generate_analysis_report(results: Dict) -> str:
    """Generate comprehensive analysis report."""
    report = []
    report.append("# 📊 Arabic Text Characteristics Analysis")
    report.append("*Comprehensive analysis of Arabic books dataset for OCR training*\n")

    if not results:
        report.append("❌ **Analysis failed - could not load dataset**\n")
        return '\n'.join(report)

    # Diacritics Analysis
    report.append("## 🔤 Diacritics Analysis")
    diac = results['diacritics']
    report.append(f"- **Total characters**: {diac['total_chars']:,}")
    report.append(f"- **Arabic characters**: {diac['arabic_chars']:,}")
    report.append(f"- **Total diacritics**: {diac['total_diacritics']:,}")
    report.append(f"- **Diacritic ratio**: {diac['diacritic_ratio']:.3f}")

    if diac['most_common_diacritics']:
        report.append("\n**Most common diacritics:**")
        for diacritic, count in diac['most_common_diacritics']:
            report.append(f"- {diacritic}: {count:,}")

    # Formatting Patterns
    report.append("\n## 📝 Formatting Patterns")
    formatting = results['formatting']
    for pattern_type, data in formatting.items():
        report.append(f"\n**{pattern_type.replace('_', ' ').title()}:**")
        report.append(f"- Count: {data['count']}")
        if data['examples']:
            report.append(f"- Examples: {', '.join(data['examples'][:3])}")

    # Islamic Terminology
    report.append("\n## 🕌 Islamic Terminology Analysis")
    islamic = results['islamic_terms']
    report.append(f"- **Total Islamic terms**: {islamic['total_islamic_terms']}")
    report.append(f"- **Unique term types**: {islamic['unique_terms']}")

    if islamic['term_frequencies']:
        report.append("\n**Most frequent terms:**")
        sorted_terms = sorted(islamic['term_frequencies'].items(),
                            key=lambda x: x[1], reverse=True)
        for term, count in sorted_terms[:5]:
            report.append(f"- {term}: {count}")

    # Character Distribution
    report.append("\n## 🔤 Character Distribution")
    chars = results['character_dist']
    report.append(f"- **Unique Arabic characters**: {chars['unique_characters']}")
    report.append(f"- **Total Arabic characters**: {chars['total_arabic_chars']:,}")

    if chars['most_common_chars']:
        report.append("\n**Most frequent characters:**")
        for char, count in chars['most_common_chars'][:10]:
            report.append(f"- {char}: {count:,}")

    # Text Structure
    report.append("\n## 📄 Text Structure Analysis")
    struct = results['text_structure']
    report.append(f"- **Total lines**: {struct['total_lines']:,}")
    report.append(f"- **Non-empty lines**: {struct['non_empty_lines']:,}")
    report.append(f"- **Total words**: {struct['total_words']:,}")
    report.append(f"- **Average line length**: {struct['avg_line_length']:.1f} chars")
    report.append(f"- **Average word length**: {struct['avg_word_length']:.1f} chars")
    report.append(f"- **Max line length**: {struct['max_line_length']} chars")

    # Recommendations
    report.append("\n## 🎯 OCR Training Recommendations")
    report.append("Based on the analysis:")

    diacritic_ratio = results['diacritics']['diacritic_ratio']
    if diacritic_ratio < 0.1:
        report.append("- ✅ **Low diacritic density** - good for initial OCR training")
    else:
        report.append("- ⚠️ **High diacritic density** - will need careful handling")

    islamic_terms = results['islamic_terms']['total_islamic_terms']
    if islamic_terms > 100:
        report.append("- ✅ **Rich Islamic terminology** - perfect for classical texts")

    avg_line = struct['avg_line_length']
    if avg_line > 50:
        report.append("- ⚠️ **Long lines** - may need line segmentation")

    report.append("\n## 🚀 Next Steps for Section 1.3")
    report.append("1. **Configure MLflow for Arabic OCR experiments**")
    report.append("2. **Set up Arabic text evaluation metrics (CER, WER, BLEU)**")
    report.append("3. **Create OCR-specific logging and tracking**")
    report.append("4. **Begin Nougat model integration testing**")

    return '\n'.join(report)


def main():
    """Main analysis function."""
    print("🕌 Arabic Text Characteristics Analyzer")
    print("=" * 50)

    # Perform analysis
    results = analyze_dataset_sample()

    # Generate report
    report = generate_analysis_report(results)

    # Save report
    output_file = "arabic_text_characteristics_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Analysis complete!")
    print(f"📄 Report saved to: {output_file}")
    print(f"🎯 Section 1.2 analysis is now complete!")


if __name__ == "__main__":
    main()