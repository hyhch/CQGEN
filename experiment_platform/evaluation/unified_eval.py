"""Unified SBERT-based evaluation for CQ generation methods.

Uses set-based cosine-similarity matching with the all-MiniLM-L6-v2 model
for fair cross-method comparison.

Matching semantics (set-based / many-to-many):
  - A generated CQ counts as a true positive if it is semantically similar
    (cosine similarity > threshold) to ANY ground truth CQ.
  - A ground truth CQ counts as covered if ANY generated CQ is similar to it.
  Precision = |{gen CQ with >= 1 GT match}| / |gen CQs|
  Recall    = |{GT CQ with >= 1 gen match}| / |GT CQs|

This is appropriate for CQ generation where multiple generated questions may
legitimately address the same ground truth concept, and we want to measure
both relevance (precision) and coverage (recall).
"""

from sentence_transformers import SentenceTransformer, util

# Lazy-loaded model singleton
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def deduplicate_cqs(cqs, dedup_threshold=0.85):
    """Remove semantically duplicate CQs using SBERT cosine similarity.

    For each pair of CQs with similarity >= dedup_threshold, the later one
    is dropped. Returns (deduplicated_cqs, original_count).
    """
    if len(cqs) <= 1:
        return list(cqs), len(cqs)

    model = _get_model()
    embeddings = model.encode(cqs, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)

    keep = []
    removed = set()
    for i in range(len(cqs)):
        if i in removed:
            continue
        keep.append(cqs[i])
        # Mark all later CQs that are too similar as duplicates
        for j in range(i + 1, len(cqs)):
            if j not in removed and cos_scores[i][j].item() >= dedup_threshold:
                removed.add(j)

    return keep, len(cqs)


def evaluate(generated_cqs, ground_truth_cqs, threshold=0.6,
             dedup=False, dedup_threshold=0.85, gt_labels=None):
    """Compute P/R/F1 via set-based (many-to-many) cosine-similarity matching.

    A generated CQ is a true positive if similar to ANY ground truth CQ.
    A ground truth CQ is covered if ANY generated CQ is similar to it.

    Args:
        generated_cqs: List of generated CQ strings.
        ground_truth_cqs: List of ground truth CQ strings.
        threshold: Cosine similarity threshold for a match.
        dedup: If True, deduplicate generated CQs before evaluation.
        dedup_threshold: Cosine similarity threshold for deduplication.
        gt_labels: Optional dict mapping GT CQ string -> "Simple"/"Complex".

    Returns:
        Dict with precision, recall, f1, matched_count, per_cq_scores,
        and optionally recall_simple, recall_complex if gt_labels provided.
    """
    original_count = len(generated_cqs)

    # Optional deduplication
    if dedup and generated_cqs:
        generated_cqs, _ = deduplicate_cqs(generated_cqs, dedup_threshold)

    if not generated_cqs or not ground_truth_cqs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_count": 0,
            "generated_count": len(generated_cqs),
            "generated_count_before_dedup": original_count,
            "ground_truth_count": len(ground_truth_cqs),
            "per_cq_scores": [],
        }

    model = _get_model()
    gt_emb = model.encode(ground_truth_cqs, convert_to_tensor=True)
    pred_emb = model.encode(generated_cqs, convert_to_tensor=True)
    cosine_scores = util.cos_sim(pred_emb, gt_emb)  # shape: (pred, gt)

    # Set-based (many-to-many) matching
    matched_gt = set()
    matched_pred = set()
    for i in range(len(generated_cqs)):
        for j in range(len(ground_truth_cqs)):
            if cosine_scores[i][j].item() > threshold:
                matched_pred.add(i)
                matched_gt.add(j)

    precision = len(matched_pred) / len(generated_cqs) if generated_cqs else 0
    recall = len(matched_gt) / len(ground_truth_cqs) if ground_truth_cqs else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Per generated-CQ: best similarity to any ground truth CQ
    per_cq_scores = []
    for i in range(len(generated_cqs)):
        best_score = max(cosine_scores[i][j].item() for j in range(len(ground_truth_cqs)))
        best_gt_idx = max(range(len(ground_truth_cqs)),
                          key=lambda j: cosine_scores[i][j].item())
        per_cq_scores.append({
            "cq": generated_cqs[i],
            "best_similarity": round(best_score, 4),
            "best_gt": ground_truth_cqs[best_gt_idx],
            "matched": i in matched_pred,
        })

    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_count": len(matched_gt),
        "matched_pred_count": len(matched_pred),
        "generated_count": len(generated_cqs),
        "generated_count_before_dedup": original_count,
        "ground_truth_count": len(ground_truth_cqs),
        "per_cq_scores": per_cq_scores,
    }

    # Simple/Complex recall breakdown
    if gt_labels:
        labels_per_idx = [
            gt_labels.get(cq.strip(), "unknown") for cq in ground_truth_cqs
        ]
        simple_total = sum(1 for l in labels_per_idx if l.lower() == "simple")
        complex_total = sum(1 for l in labels_per_idx if l.lower() == "complex")
        simple_matched = sum(
            1 for idx in matched_gt
            if labels_per_idx[idx].lower() == "simple"
        )
        complex_matched = sum(
            1 for idx in matched_gt
            if labels_per_idx[idx].lower() == "complex"
        )
        result["recall_simple"] = round(simple_matched / simple_total, 4) if simple_total > 0 else 0.0
        result["recall_complex"] = round(complex_matched / complex_total, 4) if complex_total > 0 else 0.0
        result["simple_total"] = simple_total
        result["simple_matched"] = simple_matched
        result["complex_total"] = complex_total
        result["complex_matched"] = complex_matched

    return result
