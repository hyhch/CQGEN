#!/usr/bin/env python3
"""RETROFIT-CQs: One-command pipeline for retrofitting Competency Questions from ontologies."""

import logging
import os
import sys

from dotenv import load_dotenv

from config import (
    parse_args,
    detect_ontology_format,
    create_run_dir,
    setup_logging,
    save_run_config,
)

logger = logging.getLogger(__name__)


def main(argv=None):
    load_dotenv()
    args = parse_args(argv)

    # --- Create run directory and set up logging ---
    run_dir = create_run_dir(args.output_dir)
    setup_logging(run_dir)
    save_run_config(run_dir, args)

    logger.info("=" * 60)
    logger.info("RETROFIT-CQs Pipeline")
    logger.info("Run directory: %s", run_dir)
    logger.info("Ontology: %s", args.ontology)
    logger.info("LLM backend: %s", args.llm)
    logger.info("=" * 60)

    # --- Stage 1: Triple extraction ---
    triples_csv = os.path.join(run_dir, "triples.csv")

    if args.skip_extract:
        if args.triples_csv:
            triples_csv = args.triples_csv
            logger.info("Stage 1 SKIPPED — using existing triples: %s", triples_csv)
        else:
            logger.error("--skip-extract requires --triples-csv")
            sys.exit(1)
    else:
        from extract_triples import extract_triples_rdflib, extract_triples_deeponto

        logger.info("--- Stage 1: Triple Extraction ---")
        if args.use_deeponto:
            extract_triples_deeponto(args.ontology, triples_csv)
        else:
            fmt = detect_ontology_format(args.ontology)
            extract_triples_rdflib(args.ontology, triples_csv, fmt)

    # --- Stage 2: CQ generation ---
    generated_xlsx = os.path.join(run_dir, "generated_cqs.xlsx")

    if args.skip_generate:
        if args.generated_cqs:
            generated_xlsx = args.generated_cqs
            logger.info("Stage 2 SKIPPED — using existing CQs: %s", generated_xlsx)
        else:
            logger.error("--skip-generate requires --generated-cqs")
            sys.exit(1)
    else:
        from generate_cqs import create_llm_client, generate_questions_from_csv

        logger.info("--- Stage 2: CQ Generation ---")
        client = create_llm_client(args.llm)
        generate_questions_from_csv(client, triples_csv, generated_xlsx)

    # --- Stage 3: Evaluation ---
    from evaluate_cqs import (
        load_benchmark_cqs,
        extract_generated_cqs,
        run_sbert_matching,
        compute_metrics,
    )

    logger.info("--- Stage 3: Evaluation ---")
    matched_xlsx = os.path.join(run_dir, "matched_cqs.xlsx")
    metrics_json = os.path.join(run_dir, "metrics.json")

    benchmark_cqs = load_benchmark_cqs(
        args.benchmark,
        ontology_path=args.ontology,
        benchmark_filter=args.benchmark_filter,
    )
    generated_cqs = extract_generated_cqs(generated_xlsx)

    matched_count, precision_numerator, _ = run_sbert_matching(
        benchmark_cqs, generated_cqs, args.threshold, matched_xlsx
    )
    metrics = compute_metrics(
        len(benchmark_cqs), matched_count, precision_numerator,
        len(generated_cqs), metrics_json
    )

    # --- Optional: CQ labeling ---
    if args.label:
        from generate_cqs import create_llm_client as _create_client
        from label_cqs import label_cqs_file

        logger.info("--- Optional: CQ Labeling ---")
        label_client = _create_client(args.llm)
        labeled_xlsx = os.path.join(run_dir, "labeled_cqs.xlsx")
        label_cqs_file(label_client, matched_xlsx, labeled_xlsx)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Benchmark CQs:  %d", metrics["benchmark_cqs"])
    logger.info("  Generated CQs:  %d", metrics["generated_cqs"])
    logger.info("  Precision:      %.4f", metrics["precision"])
    logger.info("  Recall:         %.4f", metrics["recall"])
    logger.info("  F1 Score:       %.4f", metrics["f1_score"])
    logger.info("  Output dir:     %s", run_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
