import json
import logging
import os
import re

import pandas as pd
from rdflib import Graph, OWL, RDF

logger = logging.getLogger(__name__)

SBERT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


# ---------------------------------------------------------------------------
# Benchmark CQ loading with auto-detection
# ---------------------------------------------------------------------------

def _detect_ontology_uri(ontology_path):
    """Extract the ontology URI from an ontology file using rdflib."""
    from config import detect_ontology_format

    fmt = detect_ontology_format(ontology_path)
    g = Graph()
    try:
        g.parse(ontology_path, format=fmt)
    except Exception as e:
        logger.warning("Could not parse ontology for URI detection: %s", e)
        return None

    for s in g.subjects(RDF.type, OWL.Ontology):
        uri = str(s)
        logger.info("Detected ontology URI: %s", uri)
        return uri
    # Fallback: try base namespace
    for ns_prefix, ns_uri in g.namespaces():
        if ns_prefix == "":
            uri = str(ns_uri)
            logger.info("Using base namespace as ontology URI: %s", uri)
            return uri
    return None


def load_benchmark_cqs(benchmark_path, ontology_path=None, benchmark_filter=None):
    """Load benchmark CQs from Excel, optionally filtering by ontology.

    Auto-detects the ontology URI from the ontology file and matches it
    against the Ontology column in the benchmark file.

    Returns:
        List of benchmark CQ strings.
    """
    logger.info("Loading benchmark CQs from %s", benchmark_path)
    df = pd.read_excel(benchmark_path)

    # Determine filter
    filter_str = benchmark_filter
    if filter_str is None and ontology_path is not None:
        detected_uri = _detect_ontology_uri(ontology_path)
        if detected_uri:
            # Try substring match against Ontology column values
            unique_onts = df["Ontology"].dropna().unique()
            for ont_val in unique_onts:
                # Check if the detected URI is a substring of the benchmark value or vice versa
                if detected_uri in str(ont_val) or str(ont_val) in detected_uri:
                    filter_str = str(ont_val)
                    logger.info("Auto-matched benchmark filter: %s", filter_str)
                    break

    if filter_str:
        mask = df["Ontology"].astype(str).str.contains(filter_str, na=False)
        df = df[mask]
        logger.info("Filtered to %d benchmark CQs (filter=%s)", len(df), filter_str)
    else:
        logger.warning("No ontology filter applied — using all %d benchmark CQs", len(df))

    cqs = df["Competency Questions"].dropna().astype(str).tolist()
    return cqs


# ---------------------------------------------------------------------------
# Generated CQ extraction
# ---------------------------------------------------------------------------

def extract_generated_cqs(generated_path):
    """Extract individual CQs from a generated CQs Excel file.

    Handles multi-line CQ answers by splitting on newlines.

    Returns:
        List of individual CQ strings.
    """
    logger.info("Extracting generated CQs from %s", generated_path)
    df = pd.read_excel(generated_path)

    # Try 'Question' column first (our format), then 'Sentence2' (sbert format)
    col = None
    for candidate in ["Question", "Sentence2"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        # Use the last column as fallback
        col = df.columns[-1]
        logger.warning("Using last column '%s' as CQ source", col)

    cqs = []
    for cell in df[col].dropna().astype(str):
        for line in cell.splitlines():
            line = line.strip()
            # Remove numbering prefix like "1. ", "2. "
            line = re.sub(r"^\d+\.\s*", "", line)
            if line and line != "Error generating question":
                cqs.append(line)

    logger.info("Extracted %d individual generated CQs", len(cqs))
    return cqs


# ---------------------------------------------------------------------------
# SBERT matching (reproduces sbert_final_2.py logic)
# ---------------------------------------------------------------------------

def run_sbert_matching(benchmark_cqs, generated_cqs, threshold, output_xlsx):
    """Match benchmark CQs against generated CQs using SBERT.

    For each benchmark CQ, finds generated CQs with cosine similarity >= threshold.
    Matched generated CQs are removed from the pool to prevent double-counting.

    Returns:
        (matched_count, output_rows) where matched_count is the number of
        benchmark CQs that had at least one match.
    """
    from sentence_transformers import SentenceTransformer, util as st_util

    logger.info("Loading SBERT model: %s", SBERT_MODEL)
    model = SentenceTransformer(SBERT_MODEL)

    # Pre-encode all generated CQs
    logger.info("Encoding %d generated CQs", len(generated_cqs))
    gen_embeddings = model.encode(generated_cqs, convert_to_tensor=True)

    pool = list(range(len(generated_cqs)))  # indices into generated_cqs
    output_rows = []
    matched_count = 0
    precision_numerator = 0

    logger.info("Matching %d benchmark CQs (threshold=%.2f)", len(benchmark_cqs), threshold)
    for i, bench_cq in enumerate(benchmark_cqs):
        bench_emb = model.encode(bench_cq, convert_to_tensor=True)
        matched = []
        remaining = []

        for idx in pool:
            sim = st_util.pytorch_cos_sim(bench_emb, gen_embeddings[idx]).item()
            if sim >= threshold:
                matched.append(generated_cqs[idx])
            else:
                remaining.append(idx)

        pool = remaining
        decision = ", ".join(matched) if matched else "No match"
        output_rows.append([bench_cq, decision])

        if matched:
            matched_count += 1
            precision_numerator += len(matched)

        if (i + 1) % 10 == 0:
            logger.info("  Processed %d/%d benchmark CQs", i + 1, len(benchmark_cqs))

    # Save matched results
    output_df = pd.DataFrame(output_rows, columns=["Sentence1", "Decision"])
    output_df.to_excel(output_xlsx, index=False)
    logger.info("Matching results saved to %s", output_xlsx)

    return matched_count, precision_numerator, output_rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(benchmark_count, matched_count, precision_numerator, total_generated, output_path):
    """Compute precision, recall, F1 and save to JSON.

    precision = matched_generated_cqs / total_generated_cqs
    recall = matched_benchmark_cqs / total_benchmark_cqs
    """
    precision = precision_numerator / total_generated if total_generated > 0 else 0
    recall = matched_count / benchmark_count if benchmark_count > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "benchmark_cqs": benchmark_count,
        "generated_cqs": total_generated,
        "matched_benchmark_cqs": matched_count,
        "matched_generated_cqs": precision_numerator,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics: precision=%.4f  recall=%.4f  F1=%.4f", precision, recall, f1)
    logger.info("Metrics saved to %s", output_path)
    return metrics
