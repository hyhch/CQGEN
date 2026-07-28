#!/usr/bin/env python3
"""Batch experiment runner for the OntologyAgent CQ Retrofitting pipeline.

Usage examples:
    # Single experiment with Config.py defaults
    python run_experiments.py

    # Specific dataset + LLM
    python run_experiments.py --dataset demcare --model qwen-max

    # All datasets x all configured LLMs
    python run_experiments.py --all

    # All datasets with the current (default) LLM
    python run_experiments.py --all-datasets

    # Ablation: compare all 4 segmentation methods on one dataset
    python run_experiments.py --ablation segmentation --dataset demcare

    # Ablation: segmentation comparison across all datasets
    python run_experiments.py --ablation segmentation --all-datasets

    # Use a specific segmentation method
    python run_experiments.py --segmentation louvain

    # Specific random seed
    python run_experiments.py --seed 123

    # Combine flags freely
    python run_experiments.py --dataset onem2m --model qwen-max --segmentation leiden --seed 99
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime

from config.Config import (
    DATA_DIR_PATH,
    ONTOLOGY_NAME,
    CQ_EXAMPLES_NUM,
    MAX_LOOP_COUNT,
    MAX_NUM_SUB_TRIPLES,
    SEGMENTATION_METHOD,
    RANDOM_SEED,
    LLM_API_KEY,
    LLM_API_TYPE,
    LLM_BASE_URL,
    LLM_MODEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "demcare",
    "onem2m",
    "saref4env",
    "vicinitycore",
    "videogameontology",
]

ALL_LLMS = {
    "qwen-max": {
        "api_type": "openai",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "YOUR_API_KEY_HERE",
    },
    # --- Add more LLMs here as needed ---
    # "gpt-4o": {
    #     "api_type": "openai",
    #     "model": "gpt-4o",
    #     "base_url": "https://api.openai.com/v1",
    #     "api_key": "sk-YOUR_OPENAI_KEY",
    # },
    # "llama3": {
    #     "api_type": "ollama",
    #     "model": "llama3",
    #     "base_url": "http://localhost:11434/v1",
    #     "api_key": "ollama",
    # },
    # "qwen-plus": {
    #     "api_type": "openai",
    #     "model": "qwen-plus",
    #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     "api_key": "sk-YOUR_DASHSCOPE_KEY",
    # },
}

ALL_SEGMENTATION_METHODS = ["auto", "metis", "louvain", "leiden", "spectral"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_config(
    api_type: str, model: str, base_url: str, api_key: str
) -> dict:
    """Build the llm_config dict expected by ``cq_main``."""
    return {
        "api_type": api_type,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _print_header(
    dataset: str, model: str, segmentation: str, seed: int, index: int, total: int
) -> None:
    """Print a clear experiment header."""
    border = "#" * 80
    print(f"\n{border}")
    print(f"#  EXPERIMENT {index}/{total}")
    print(f"#  Dataset:       {dataset}")
    print(f"#  LLM:           {model}")
    print(f"#  Segmentation:  {segmentation}")
    print(f"#  Seed:          {seed}")
    print(f"#  Started at:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{border}\n")


def _print_summary(results: list[dict]) -> None:
    """Print a summary table of all completed experiments."""
    border = "=" * 100
    print(f"\n{border}")
    print("  EXPERIMENT SUMMARY")
    print(f"{border}")
    header = (
        f"  {'#':<4} {'Dataset':<22} {'LLM':<16} {'Segmentation':<14} "
        f"{'Seed':<6} {'Status':<10} {'Duration':<12}"
    )
    print(header)
    print(f"  {'-' * 94}")
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        duration = _format_duration(r["duration"])
        print(
            f"  {r['index']:<4} {r['dataset']:<22} {r['model']:<16} "
            f"{r['segmentation']:<14} {r['seed']:<6} {status:<10} {duration:<12}"
        )
    print(f"{border}")

    succeeded = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    total_time = sum(r["duration"] for r in results)
    print(
        f"  Total: {len(results)} experiments | "
        f"{succeeded} succeeded | {failed} failed | "
        f"Total time: {_format_duration(total_time)}"
    )
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


async def run_single_experiment(
    dataset: str,
    llm_config: dict,
    segmentation: str,
    seed: int,
    data_dir: str,
    results_dir: str,
    index: int,
    total: int,
) -> dict:
    """Run one experiment and return a result dict."""
    # Lazy import so argparse --help does not require metagpt
    from src.CQRetrofit import main as cq_main

    model_name = llm_config["model"]
    _print_header(dataset, model_name, segmentation, seed, index, total)

    record = {
        "index": index,
        "dataset": dataset,
        "model": model_name,
        "segmentation": segmentation,
        "seed": seed,
        "success": False,
        "duration": 0.0,
    }

    t0 = time.time()
    try:
        await cq_main(
            data_dir=data_dir,
            ontology_name=dataset,
            cq_examples_num=CQ_EXAMPLES_NUM,
            max_loop_count=MAX_LOOP_COUNT,
            max_sub_triples=MAX_NUM_SUB_TRIPLES,
            segmentation_method=segmentation,
            random_seed=seed,
            results_dir=results_dir,
            llm_config=llm_config,
        )
        record["success"] = True
    except Exception as exc:
        print(f"\n[ERROR] Experiment {index}/{total} failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        record["duration"] = time.time() - t0

    status_msg = "SUCCEEDED" if record["success"] else "FAILED"
    print(
        f"\n>>> Experiment {index}/{total} {status_msg} "
        f"in {_format_duration(record['duration'])}\n"
    )
    return record


def _build_experiment_list(args) -> list[dict]:
    """Build the list of experiment configurations from parsed CLI args.

    Each entry is a dict with keys: dataset, llm_config, segmentation, seed.
    """
    experiments: list[dict] = []

    # --- Determine datasets ------------------------------------------------
    if args.all or args.all_datasets:
        datasets = list(ALL_DATASETS)
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [ONTOLOGY_NAME]  # Config.py default

    # --- Determine LLMs ---------------------------------------------------
    if args.all:
        llm_configs = {name: cfg for name, cfg in ALL_LLMS.items()}
    elif args.model:
        if args.model in ALL_LLMS:
            llm_configs = {args.model: ALL_LLMS[args.model]}
        else:
            # Build config from defaults, overriding only the model name
            llm_configs = {
                args.model: _make_llm_config(
                    api_type=LLM_API_TYPE,
                    model=args.model,
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                )
            }
    else:
        # Use the default LLM from Config.py
        llm_configs = {
            LLM_MODEL: _make_llm_config(
                api_type=LLM_API_TYPE,
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
            )
        }

    # --- Determine segmentation methods -----------------------------------
    if args.ablation == "segmentation":
        segmentation_methods = list(ALL_SEGMENTATION_METHODS)
    elif args.segmentation:
        segmentation_methods = [args.segmentation]
    else:
        segmentation_methods = [SEGMENTATION_METHOD]  # Config.py default

    # --- Determine seed ---------------------------------------------------
    seed = args.seed if args.seed is not None else RANDOM_SEED

    # --- Cartesian product ------------------------------------------------
    for ds in datasets:
        for _llm_name, llm_cfg in llm_configs.items():
            for seg in segmentation_methods:
                experiments.append(
                    {
                        "dataset": ds,
                        "llm_config": llm_cfg,
                        "segmentation": seg,
                        "seed": seed,
                    }
                )

    return experiments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for OntologyAgent CQ Retrofitting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset selection
    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--dataset",
        type=str,
        choices=ALL_DATASETS,
        help="Run on a single dataset (default: from Config.py).",
    )
    ds_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run on all available datasets with the selected LLM(s).",
    )
    ds_group.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets x all configured LLMs (full grid).",
    )

    # LLM selection
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model name (must match a key in ALL_LLMS, or will use default provider settings).",
    )

    # Segmentation
    parser.add_argument(
        "--segmentation",
        type=str,
        choices=ALL_SEGMENTATION_METHODS,
        help="Segmentation algorithm to use (default: from Config.py).",
    )

    # Ablation
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["segmentation"],
        help="Run ablation study. 'segmentation' compares all 4 methods.",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: from Config.py).",
    )

    # Directories
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR_PATH,
        help=f"Path to the dataset directory (default: {DATA_DIR_PATH}).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to store experiment results (default: results).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    experiments = _build_experiment_list(args)

    if not experiments:
        print("No experiments to run. Check your arguments.")
        sys.exit(1)

    total = len(experiments)
    print(f"\n{'=' * 80}")
    print(f"  OntologyAgent Batch Experiment Runner")
    print(f"  {total} experiment(s) queued")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # Preview the experiment plan
    for idx, exp in enumerate(experiments, 1):
        print(
            f"  [{idx}/{total}] dataset={exp['dataset']}  "
            f"model={exp['llm_config']['model']}  "
            f"seg={exp['segmentation']}  seed={exp['seed']}"
        )
    print()

    results: list[dict] = []

    for idx, exp in enumerate(experiments, 1):
        record = await run_single_experiment(
            dataset=exp["dataset"],
            llm_config=exp["llm_config"],
            segmentation=exp["segmentation"],
            seed=exp["seed"],
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            index=idx,
            total=total,
        )
        results.append(record)

    _print_summary(results)

    # Exit with non-zero if any experiment failed
    if any(not r["success"] for r in results):
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
