"""Evaluation: compare generated CQs against ground truth using semantic similarity."""

import logging
import os
import re

import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer, util

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.6

# Lazy-loaded model singleton
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _load_ground_truth(input_path):
    """Load ground truth CQs from input_path/cqs/cqs.yml."""
    with open(os.path.join(input_path, 'cqs', 'cqs.yml')) as f:
        data = yaml.safe_load(f)
    return [c['question'] for c in data['ontology']['cqs']]


def _normalize_text(text):
    """Normalize: lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_similarity(gt_cqs, predictions, threshold=SIMILARITY_THRESHOLD):
    """Compute P/R/F1 via greedy cosine-similarity matching.

    Returns (precision, recall, f1, matched_gt_indices, matched_pred_indices).
    """
    if not predictions:
        return 0, 0, 0, set(), set()

    model = _get_model()
    gt_emb = model.encode(gt_cqs, convert_to_tensor=True)
    pred_emb = model.encode(predictions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(gt_emb, pred_emb)

    # Build all pairs sorted by score descending
    pairs = []
    for i in range(len(cosine_scores)):
        for j in range(len(cosine_scores[0])):
            pairs.append((i, j, cosine_scores[i][j].item()))
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Greedy 1-to-1 matching
    matched_gt = set()
    matched_pred = set()
    for i, j, score in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        if score > threshold:
            matched_gt.add(i)
            matched_pred.add(j)

    precision = len(matched_pred) / len(predictions)
    recall = len(matched_gt) / len(gt_cqs) if gt_cqs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, matched_gt, matched_pred


def compute_classification_stats(gt_cqs, matched_gt_indices, labels_path):
    """Compute simple/complex hit rates using CQ labels from an Excel file.

    Returns dict with hit rates, or None if labels_path is invalid.
    """
    if not labels_path or not os.path.exists(labels_path):
        return None

    df = pd.read_excel(labels_path)
    if 'Competency Questions' not in df.columns or 'label' not in df.columns:
        log.warning("Labels file missing required columns: 'Competency Questions', 'label'")
        return None

    label_map = dict(zip(
        df['Competency Questions'].apply(_normalize_text),
        df['label'],
    ))

    labels = [label_map.get(_normalize_text(cq), 'unknown') for cq in gt_cqs]

    simple_total = sum(1 for l in labels if isinstance(l, str) and l.lower() == 'simple')
    complex_total = sum(1 for l in labels if isinstance(l, str) and l.lower() == 'complex')
    simple_hit = sum(1 for idx in matched_gt_indices
                     if idx < len(labels) and isinstance(labels[idx], str) and labels[idx].lower() == 'simple')
    complex_hit = sum(1 for idx in matched_gt_indices
                      if idx < len(labels) and isinstance(labels[idx], str) and labels[idx].lower() == 'complex')

    return {
        'simple_hit_rate': simple_hit / simple_total if simple_total > 0 else 0,
        'complex_hit_rate': complex_hit / complex_total if complex_total > 0 else 0,
        'simple_total': simple_total,
        'complex_total': complex_total,
    }


def evaluate_file(gt_cqs, prediction_path, threshold=SIMILARITY_THRESHOLD, labels_path=None):
    """Evaluate a single prediction file against ground truth CQs."""
    with open(prediction_path) as f:
        predictions = [line.strip() for line in f if line.strip()]

    if not predictions:
        log.warning("No predictions in %s", prediction_path)
        return {'precision': 0, 'recall': 0, 'f1_score': 0}

    precision, recall, f1, matched_gt, _ = compute_similarity(gt_cqs, predictions, threshold)
    result = {'precision': precision, 'recall': recall, 'f1_score': f1}

    if labels_path:
        stats = compute_classification_stats(gt_cqs, matched_gt, labels_path)
        if stats:
            result.update(stats)

    return result


def evaluate_ontology(onto_name, threshold=SIMILARITY_THRESHOLD,
                      pred_root='data_out', data_root='data', labels_path=None):
    """Evaluate all prediction files for a given ontology."""
    input_path = os.path.join(data_root, onto_name)
    gt_cqs = _load_ground_truth(input_path)
    onto_pred_dir = os.path.join(pred_root, onto_name)

    if not os.path.isdir(onto_pred_dir):
        log.warning("No predictions directory: %s", onto_pred_dir)
        return []

    scores = []
    for mode in sorted(os.listdir(onto_pred_dir)):
        mode_dir = os.path.join(onto_pred_dir, mode)
        if mode == 'archive' or not os.path.isdir(mode_dir):
            continue
        for filename in sorted(os.listdir(mode_dir)):
            if not filename.endswith('.txt') or filename.endswith('_raw.json'):
                continue

            result = evaluate_file(
                gt_cqs,
                os.path.join(mode_dir, filename),
                threshold,
                labels_path,
            )

            # Parse LLM name and example count from filename: <onto>_<llm>_<examples>.txt
            parts = filename.replace('.txt', '').split('_')
            llm_name = parts[1] if len(parts) > 1 else 'unknown'
            examples = parts[2] if len(parts) > 2 else '0'

            scores.append({
                'onto': onto_name,
                'llm': llm_name,
                'mode': mode,
                'examples': examples,
                **result,
            })

    return scores


def evaluate_all(threshold=SIMILARITY_THRESHOLD,
                 pred_root='data_out', data_root='data', labels_path=None):
    """Evaluate all ontologies found in pred_root."""
    scores = []
    for onto in sorted(os.listdir(pred_root)):
        if os.path.isdir(os.path.join(pred_root, onto)):
            scores.extend(evaluate_ontology(onto, threshold, pred_root, data_root, labels_path))
    return scores
