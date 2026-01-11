"""
Evaluation metrics for Arabic OCR tasks.

Provides Character Error Rate (CER), Word Error Rate (WER), and BLEU score
calculations for OCR model evaluation.
"""

import re
from typing import List, Tuple
from difflib import SequenceMatcher


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for fair comparison.

    Args:
        text: Input Arabic text

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Basic Arabic normalization (can be extended)
    normalizations = [
        ('\u0622', '\u0627'),  # Alef with madda -> alef
        ('\u0623', '\u0627'),  # Alef with hamza above -> alef
        ('\u0625', '\u0627'),  # Alef with hamza below -> alef
        ('\u0649', '\u064A'),  # Alef maksura -> yeh
        ('\u0629', '\u0647'),  # Teh marbuta -> heh
        ('\u0640', ''),        # Remove tatweel
    ]

    for old_char, new_char in normalizations:
        text = text.replace(old_char, new_char)

    return text


def calculate_cer(predicted: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Calculate Character Error Rate (CER).

    CER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total characters

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text
        normalize: Whether to normalize Arabic text before comparison

    Returns:
        CER as a float between 0.0 and 1.0
    """
    if normalize:
        predicted = normalize_arabic_text(predicted)
        ground_truth = normalize_arabic_text(ground_truth)

    if len(ground_truth) == 0:
        return 1.0 if len(predicted) > 0 else 0.0

    # Use difflib to compute edit distance
    matcher = SequenceMatcher(None, ground_truth, predicted)

    # Count operations
    substitutions = 0
    deletions = 0
    insertions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            deletions += i2 - i1
        elif tag == 'insert':
            insertions += j2 - j1

    total_chars = len(ground_truth)
    cer = (substitutions + deletions + insertions) / total_chars

    return min(cer, 1.0)  # Cap at 1.0


def calculate_wer(predicted: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total words

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text
        normalize: Whether to normalize Arabic text before comparison

    Returns:
        WER as a float between 0.0 and 1.0
    """
    if normalize:
        predicted = normalize_arabic_text(predicted)
        ground_truth = normalize_arabic_text(ground_truth)

    # Split into words
    pred_words = predicted.split()
    gt_words = ground_truth.split()

    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0

    # Use difflib to compute edit distance on word level
    matcher = SequenceMatcher(None, gt_words, pred_words)

    # Count word-level operations
    substitutions = 0
    deletions = 0
    insertions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            deletions += i2 - i1
        elif tag == 'insert':
            insertions += j2 - j1

    total_words = len(gt_words)
    wer = (substitutions + deletions + insertions) / total_words

    return min(wer, 1.0)  # Cap at 1.0


def calculate_bleu(predicted: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Calculate BLEU score (simplified version for single reference).

    Args:
        predicted: Predicted text from OCR model
        ground_truth: Ground truth text (single reference)
        normalize: Whether to normalize Arabic text before comparison

    Returns:
        BLEU score as a float between 0.0 and 1.0
    """
    if normalize:
        predicted = normalize_arabic_text(predicted)
        ground_truth = normalize_arabic_text(ground_truth)

    # Split into words
    pred_words = predicted.split()
    gt_words = ground_truth.split()

    if len(pred_words) == 0 or len(gt_words) == 0:
        return 0.0

    # Calculate n-gram precisions (1-gram to 4-gram)
    max_n = 4
    precisions = []

    for n in range(1, max_n + 1):
        pred_ngrams = [tuple(pred_words[i:i+n]) for i in range(len(pred_words) - n + 1)]
        gt_ngrams = [tuple(gt_words[i:i+n]) for i in range(len(gt_words) - n + 1)]

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        # Count matches
        gt_counts = {}
        for ngram in gt_ngrams:
            gt_counts[ngram] = gt_counts.get(ngram, 0) + 1

        matches = 0
        for ngram in pred_ngrams:
            if ngram in gt_counts and gt_counts[ngram] > 0:
                matches += 1
                gt_counts[ngram] -= 1

        precision = matches / len(pred_ngrams)
        precisions.append(precision)

    # Brevity penalty
    brevity_penalty = min(1.0, len(pred_words) / len(gt_words))

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    import math
    log_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_sum / len(precisions))

    bleu = brevity_penalty * geometric_mean
    return bleu


def evaluate_batch(predictions: List[str], ground_truths: List[str]) -> dict:
    """
    Evaluate a batch of predictions against ground truth.

    Args:
        predictions: List of predicted texts
        ground_truths: List of ground truth texts

    Returns:
        Dictionary with average CER, WER, and BLEU scores
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    cers = []
    wers = []
    bleus = []

    for pred, gt in zip(predictions, ground_truths):
        cers.append(calculate_cer(pred, gt))
        wers.append(calculate_wer(pred, gt))
        bleus.append(calculate_bleu(pred, gt))

    return {
        "cer": sum(cers) / len(cers),
        "wer": sum(wers) / len(wers),
        "bleu": sum(bleus) / len(bleus),
        "exact_match": sum(1 for pred, gt in zip(predictions, ground_truths)
                          if normalize_arabic_text(pred) == normalize_arabic_text(gt)) / len(predictions)
    }


def format_evaluation_results(results: dict) -> str:
    """
    Format evaluation results for display.

    Args:
        results: Results dictionary from evaluate_batch

    Returns:
        Formatted string
    """
    return f"""
📊 OCR Evaluation Results:
  Character Error Rate (CER): {results['cer']:.3f} ({results['cer']*100:.1f}%)
  Word Error Rate (WER):      {results['wer']:.3f} ({results['wer']*100:.1f}%)
  BLEU Score:                 {results['bleu']:.3f}
  Exact Match:                {results['exact_match']:.3f} ({results['exact_match']*100:.1f}%)
"""