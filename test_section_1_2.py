#!/usr/bin/env python3
"""
Complete validation test suite for Section 1.2 - Arabic Text Analysis.

This script runs all validation tests to verify that section 1.2
is working correctly and produces expected outputs.
"""

import os
import sys
from pathlib import Path
import arabic_reshaper
from bidi.algorithm import get_display

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(title):
    """Print a colored header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_success(message):
    """Print success message."""
    print(f"{GREEN}✅ {message}{RESET}")


def print_error(message):
    """Print error message."""
    print(f"{RED}❌ {message}{RESET}")


def print_warning(message):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {message}{RESET}")


def format_arabic_text(text):
    """Format Arabic text for proper RTL display in console with reshaping and BiDi."""
    # Check if text contains Arabic characters
    has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)

    if has_arabic:
        try:
            # Step 1: Reshape Arabic text (connect letters properly)
            reshaped_text = arabic_reshaper.reshape(text)
            # Step 2: Apply bidirectional algorithm (RTL ordering)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            # Fallback to original text if reshaping fails
            print_warning(f"Arabic reshaping failed: {e}")
            return text
    else:
        return text


def print_arabic_line(label, arabic_text):
    """Print a line with Arabic text properly formatted."""
    formatted_text = format_arabic_text(arabic_text)
    print(f"{label}: {formatted_text}")


def test_basic_arabic_processing():
    """Test 1: Basic Arabic Text Processing Pipeline"""
    print_header("Test 1: Basic Arabic Text Processing")

    try:
        from arabic_text_processing import ArabicTextProcessor
        processor = ArabicTextProcessor()

        # Test 1: Basic Arabic text detection
        test_text = 'بسم الله الرحمن الرحيم'
        result1 = processor.is_arabic_text(test_text)
        print(f'Arabic detection: {result1}')
        assert result1 == True, "Arabic text detection failed"

        # Test 2: Diacritics handling
        diacritics_text = 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ'
        result2 = processor.remove_diacritics(diacritics_text)
        print_arabic_line('Remove diacritics', result2)

        result3 = processor.preserve_diacritics(diacritics_text)
        print_arabic_line('Preserve diacritics', result3)

        # Test 3: Text statistics
        stats = processor.extract_text_statistics(diacritics_text)
        print(f'Statistics: {stats}')

        # Validate expected values
        assert stats['total_chars'] == 38, f"Expected 38 chars, got {stats['total_chars']}"
        assert stats['arabic_chars'] == 35, f"Expected 35 Arabic chars, got {stats['arabic_chars']}"
        assert stats['diacritics_count'] == 15, f"Expected 15 diacritics, got {stats['diacritics_count']}"
        assert stats['words'] == 4, f"Expected 4 words, got {stats['words']}"

        print_success("Basic Arabic text processing tests passed")
        return True

    except Exception as e:
        print_error(f"Basic Arabic processing test failed: {e}")
        return False


def test_diacritics_analysis():
    """Test 2: Diacritics Analysis Verification"""
    print_header("Test 2: Diacritics Analysis")

    try:
        from arabic_text_characteristics_analysis import ArabicTextCharacteristicsAnalyzer
        analyzer = ArabicTextCharacteristicsAnalyzer()

        # Test with known diacritics
        test_text = 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ'
        print_arabic_line('Input text', test_text)
        print(f'Length: {len(test_text)}')

        # Manual count for verification
        diacritics_found = []
        for i, char in enumerate(test_text):
            if char in analyzer.diacritics:
                diacritics_found.append((char, analyzer.diacritics[char], i))

        print(f'Manual diacritic count: {len(diacritics_found)}')
        for char, name, pos in diacritics_found[:5]:  # Show first 5
            print(f'  Position {pos}: {char} ({name})')
        if len(diacritics_found) > 5:
            print(f'  ... and {len(diacritics_found) - 5} more')

        # Compare with analyzer
        result = analyzer.analyze_diacritics(test_text)
        print(f'Analyzer result: {result["diacritic_counts"]}')
        print(f'Total diacritics found: {result["total_diacritics"]}')

        # Validate
        assert len(diacritics_found) == 15, f"Expected 15 diacritics, found {len(diacritics_found)}"
        assert result['total_diacritics'] == 15, f"Analyzer found {result['total_diacritics']}, expected 15"

        expected_counts = {'Kasra': 6, 'Sukun': 2, 'Fatha': 4, 'Shadda': 3}
        for diacritic, expected_count in expected_counts.items():
            actual_count = result['diacritic_counts'].get(diacritic, 0)
            assert actual_count == expected_count, f"{diacritic}: expected {expected_count}, got {actual_count}"

        print_success("Diacritics analysis tests passed")
        return True

    except Exception as e:
        print_error(f"Diacritics analysis test failed: {e}")
        return False


def test_islamic_patterns():
    """Test 3: Islamic Pattern Detection"""
    print_header("Test 3: Islamic Pattern Detection")

    try:
        from arabic_text_characteristics_analysis import ArabicTextCharacteristicsAnalyzer
        import re

        analyzer = ArabicTextCharacteristicsAnalyzer()

        # Test text with known Islamic patterns
        test_text = 'قال رسول الله صلى الله عليه وسلم: الله تعالى رضي الله عنه'
        print_arabic_line('Test text', test_text)

        # Test Islamic terminology detection
        islamic_result = analyzer.analyze_islamic_terminology(test_text)
        print(f'Islamic terms found: {islamic_result}')

        # Test formatting patterns
        format_text = 'كتاب الطهارة باب الوضوء قال النبي'
        format_result = analyzer.analyze_formatting_patterns(format_text)
        print(f'Formatting patterns: {format_result}')

        # Verify specific patterns manually
        patterns_to_test = ['الله', 'رسول', 'قال', 'صلى الله عليه وسلم']
        for pattern in patterns_to_test:
            matches = re.findall(pattern, test_text)
            formatted_pattern = format_arabic_text(pattern)
            formatted_matches = [format_arabic_text(match) for match in matches]
            print(f'Pattern "{formatted_pattern}": {len(matches)} matches - {formatted_matches}')

        # Validate results
        assert islamic_result['total_islamic_terms'] == 10, f"Expected 10 Islamic terms, got {islamic_result['total_islamic_terms']}"
        assert islamic_result['unique_terms'] == 7, f"Expected 7 unique terms, got {islamic_result['unique_terms']}"
        assert islamic_result['term_frequencies']['الله'] == 4, "Should find 'الله' 4 times"
        assert islamic_result['term_frequencies']['رسول'] == 1, "Should find 'رسول' 1 time"

        print_success("Islamic pattern detection tests passed")
        return True

    except Exception as e:
        print_error(f"Islamic pattern test failed: {e}")
        return False


def test_complete_pipeline():
    """Test 4: Complete Analysis Pipeline"""
    print_header("Test 4: Complete Analysis Pipeline")

    try:
        from arabic_text_characteristics_analysis import analyze_dataset_sample, generate_analysis_report

        print('Running full analysis pipeline...')
        results = analyze_dataset_sample()

        if results:
            print_success('Analysis successful')
            print(f'Available result keys: {list(results.keys())}')

            # Check each component
            expected_keys = ['diacritics', 'formatting', 'islamic_terms', 'character_dist', 'text_structure']
            for key in expected_keys:
                if key in results and isinstance(results[key], dict) and results[key]:
                    print_success(f'{key}: {len(results[key])} metrics')
                else:
                    print_error(f'{key}: Missing or invalid')
                    return False

            # Test report generation
            report = generate_analysis_report(results)
            print_success(f'Report generated: {len(report)} characters')

            # Validate report content
            assert len(report) > 1000, f"Report too short: {len(report)} characters"
            assert 'Arabic Text Characteristics Analysis' in report, "Missing report title"
            assert 'Diacritics Analysis' in report, "Missing diacritics section"
            assert 'Islamic Terminology Analysis' in report, "Missing Islamic terms section"

        else:
            print_error('Analysis failed - no results returned')
            return False

        print_success("Complete pipeline tests passed")
        return True

    except Exception as e:
        print_error(f"Complete pipeline test failed: {e}")
        return False


def test_file_generation():
    """Test 5: File Generation"""
    print_header("Test 5: Report File Generation")

    try:
        # Test arabic_text_processing.py
        print("Testing arabic_text_processing.py...")
        from arabic_text_processing import test_arabic_processing, analyze_dataset_sample

        test_output = test_arabic_processing()
        assert len(test_output) > 1000, f"Test output too short: {len(test_output)}"
        assert 'Test 1' in test_output, "Missing test results"
        print_success("Arabic text processing module working")

        sample_output = analyze_dataset_sample()
        assert len(sample_output) > 500, f"Sample output too short: {len(sample_output)}"
        assert 'Statistics' in sample_output, "Missing statistics in sample"
        print_success("Dataset sample analysis working")

        # Test main analysis script
        print("Testing arabic_text_characteristics_analysis.py...")
        from arabic_text_characteristics_analysis import main

        # Remove existing report if present
        report_file = 'arabic_text_characteristics_report.md'
        if os.path.exists(report_file):
            os.remove(report_file)

        # Run main function
        main()

        # Check if file was created
        assert os.path.exists(report_file), f"Report file {report_file} was not created"

        # Check file content
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert len(content) > 1000, f"Report file too short: {len(content)} characters"
        assert 'Arabic Text Characteristics Analysis' in content, "Missing report title"
        print_success(f"Report file generated: {report_file}")

        print_success("File generation tests passed")
        return True

    except Exception as e:
        print_error(f"File generation test failed: {e}")
        return False


def test_file_existence():
    """Test 6: Verify Required Files Exist"""
    print_header("Test 6: File Existence Check")

    required_files = [
        'arabic_text_processing.py',
        'arabic_text_characteristics_analysis.py',
        'plan.md'
    ]

    generated_files = [
        'arabic_text_analysis_report.md',
        'arabic_text_characteristics_report.md'
    ]

    all_good = True

    # Check required files
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"Required file exists: {file_path}")
        else:
            print_error(f"Missing required file: {file_path}")
            all_good = False

    # Check generated files
    for file_path in generated_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print_success(f"Generated file exists: {file_path} ({size} bytes)")
        else:
            print_warning(f"Generated file missing: {file_path} (will be created)")

    return all_good


def main():
    """Run all validation tests for Section 1.2"""
    print_header("🕌 Section 1.2 Complete Validation Test Suite")
    print("This script validates that all Arabic text analysis components are working correctly.\n")

    # Track test results
    tests = [
        ("File Existence", test_file_existence),
        ("Basic Arabic Processing", test_basic_arabic_processing),
        ("Diacritics Analysis", test_diacritics_analysis),
        ("Islamic Pattern Detection", test_islamic_patterns),
        ("Complete Pipeline", test_complete_pipeline),
        ("File Generation", test_file_generation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print_error(f"{test_name} test failed")
        except Exception as e:
            print_error(f"{test_name} test crashed: {e}")

    # Final results
    print_header("🎯 Final Test Results")

    if passed == total:
        print_success(f"ALL TESTS PASSED! ({passed}/{total})")
        print_success("✨ Section 1.2 is fully validated and working correctly!")
        print("\n📋 What was verified:")
        print("  ✅ Arabic text detection and processing")
        print("  ✅ Diacritics analysis accuracy (15/15 diacritics)")
        print("  ✅ Islamic terminology recognition (10 terms)")
        print("  ✅ Text formatting pattern detection")
        print("  ✅ Complete analysis pipeline (5 components)")
        print("  ✅ Report generation (2 markdown files)")
        print("\n🚀 Ready to proceed to Section 1.3!")
    else:
        print_error(f"TESTS FAILED: {passed}/{total} passed")
        print_warning("❌ Section 1.2 has issues that need to be fixed before proceeding.")

        remaining = total - passed
        print(f"\n🔧 Fix the {remaining} failing test{'s' if remaining > 1 else ''} and run again.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)