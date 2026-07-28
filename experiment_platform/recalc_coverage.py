#!/usr/bin/env python3
"""Recalculate OntologyAgent coverage using unified methodology.

Unified methodology: exclude meta prefixes (owl:/rdf:/rdfs:/xsd:) + global dedup.

Usage:
    python recalc_coverage.py              # recalculate all
    python recalc_coverage.py --dry-run    # preview without modifying
"""

import argparse
import json
import os
import sys

# Add parent dir so we can import evaluation.coverage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.coverage import extract_entities_from_oa_triples

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def recalc_from_chunks(chunks):
    """Recalculate coverage from subgraph_chunks using global dedup + exclude meta."""
    global_entities = set()
    global_covered = set()
    subgraph_details = []

    for idx, chunk in enumerate(chunks):
        triples = chunk.get("triples", [])
        uncovered = set(chunk.get("uncovered_entities", []))
        chunk_entities = extract_entities_from_oa_triples(triples, exclude_meta=True)

        global_entities |= chunk_entities
        covered_in_chunk = chunk_entities - uncovered
        global_covered |= covered_in_chunk

        rate = (len(covered_in_chunk) / len(chunk_entities) * 100) if chunk_entities else 0
        subgraph_details.append({
            "subgraph_id": idx,
            "total_entities": len(chunk_entities),
            "covered_entities": len(covered_in_chunk),
            "coverage_rate": round(rate, 2),
        })

    total = len(global_entities)
    covered = len(global_covered & global_entities)
    rate = (covered / total * 100) if total > 0 else 0.0

    return {
        "total_subgraphs": len(chunks),
        "total_entities": total,
        "total_covered_entities": covered,
        "total_uncovered_entities": total - covered,
        "overall_coverage_rate": round(rate, 2),
        "subgraph_coverage_details": subgraph_details,
    }


def find_all_raw_results():
    """Find all raw OA results files (those with subgraph_chunks)."""
    raw_files = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        for fname in files:
            if fname != "results.json":
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "subgraph_chunks" in data:
                    raw_files.append(fpath)
            except (json.JSONDecodeError, IOError):
                continue
    return sorted(raw_files)


def find_all_summary_files():
    """Find all OA summary/ablation files that need coverage recalculation.

    These are files with metrics.coverage_stats containing subgraph_coverage_details.
    """
    summary_files = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            # Skip raw results files
            if fname == "results.json" or "subgraph_" in fname or fname in ("config.json", "segmentation.json"):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cs = data.get("metrics", {}).get("coverage_stats", {})
                if cs.get("subgraph_coverage_details"):
                    summary_files.append(fpath)
            except (json.JSONDecodeError, IOError):
                continue
    return sorted(summary_files)


def extract_dataset_model(fpath):
    """Extract dataset and model from file path."""
    rel = os.path.relpath(fpath, RESULTS_DIR)
    parts = rel.replace("\\", "/").split("/")

    # Patterns:
    # <dataset>/<model>/run_*/results.json  (raw)
    # <dataset>/OntologyAgent/<model>/run_*.json  (OA summary)
    # ablation/tableA/CQGen-MAS/<model>/<dataset>/run_*.json
    # ablation/tableA/Monolithic/<model>/<dataset>/run_*.json
    # ablation/tableB/seg_*/<model>/<dataset>/run_*.json

    if parts[0] == "ablation":
        # ablation/<table>/<variant>/<model>/<dataset>/run_*.json
        model = parts[3]
        dataset = parts[4]
    elif parts[1] == "OntologyAgent":
        dataset = parts[0]
        model = parts[2]
    elif parts[1] in ("LLM4KE", "Retrofit-CQ"):
        dataset = parts[0]
        model = parts[2]
    else:
        dataset = parts[0]
        model = parts[1]
    return dataset, model


def main():
    parser = argparse.ArgumentParser(description="Recalculate OA coverage stats")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying files")
    args = parser.parse_args()

    # ---- Step 1: Recalculate raw OA results files ----
    raw_files = find_all_raw_results()
    print(f"Found {len(raw_files)} raw OA results files\n")

    # Build mapping: (dataset, model, old_total, old_covered) -> new_coverage_stats
    raw_mapping = {}
    updated_raw = 0

    for fpath in raw_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        old_cs = data.get("overall_coverage_stats", {})
        old_total = old_cs.get("total_entities", "?")
        old_covered = old_cs.get("total_covered_entities", "?")

        chunks = data["subgraph_chunks"]
        new_cs = recalc_from_chunks(chunks)

        dataset, model = extract_dataset_model(fpath)
        rel = os.path.relpath(fpath, RESULTS_DIR)

        # Store mapping for summary file matching
        key = (dataset, model, old_total, old_covered)
        raw_mapping[key] = new_cs

        if old_total == new_cs["total_entities"] and old_covered == new_cs["total_covered_entities"]:
            print(f"  [SKIP] {rel}: unchanged ({old_total}/{old_covered})")
            continue

        print(f"  [UPDATE] {rel}:")
        print(f"    total: {old_total} -> {new_cs['total_entities']}")
        print(f"    covered: {old_covered} -> {new_cs['total_covered_entities']}")
        print(f"    rate: {old_cs.get('overall_coverage_rate', '?'):.2f}% -> {new_cs['overall_coverage_rate']:.2f}%")

        if not args.dry_run:
            data["overall_coverage_stats"] = new_cs
            # Also update per-chunk coverage_stats
            for idx, chunk in enumerate(chunks):
                if idx < len(new_cs["subgraph_coverage_details"]):
                    detail = new_cs["subgraph_coverage_details"][idx]
                    chunk["coverage_stats"] = {
                        "total_entities": detail["total_entities"],
                        "covered_entities": detail["covered_entities"],
                        "uncovered_entities_count": detail["total_entities"] - detail["covered_entities"],
                        "coverage_rate": detail["coverage_rate"],
                    }
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        updated_raw += 1

    print(f"\nRaw files: {updated_raw} updated out of {len(raw_files)}\n")

    # ---- Step 2: Update summary/ablation files ----
    summary_files = find_all_summary_files()
    print(f"Found {len(summary_files)} OA summary/ablation files\n")

    updated_summary = 0
    for fpath in summary_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cs = data["metrics"]["coverage_stats"]
        old_total = cs.get("total_entities", "?")
        old_covered = cs.get("total_covered_entities", "?")

        dataset, model = extract_dataset_model(fpath)
        rel = os.path.relpath(fpath, RESULTS_DIR)

        # Find matching raw results
        key = (dataset, model, old_total, old_covered)
        new_cs = raw_mapping.get(key)

        if new_cs is None:
            print(f"  [WARN] {rel}: no matching raw results for key={key}")
            continue

        if old_total == new_cs["total_entities"] and old_covered == new_cs["total_covered_entities"]:
            print(f"  [SKIP] {rel}: unchanged ({old_total}/{old_covered})")
            continue

        print(f"  [UPDATE] {rel}:")
        print(f"    total: {old_total} -> {new_cs['total_entities']}")
        print(f"    covered: {old_covered} -> {new_cs['total_covered_entities']}")
        print(f"    rate: {cs.get('overall_coverage_rate', '?'):.2f}% -> {new_cs['overall_coverage_rate']:.2f}%")

        if not args.dry_run:
            data["metrics"]["coverage_stats"] = new_cs
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        updated_summary += 1

    print(f"\nSummary files: {updated_summary} updated out of {len(summary_files)}\n")
    print(f"Total: {updated_raw + updated_summary} files {'would be ' if args.dry_run else ''}updated")


if __name__ == "__main__":
    main()
