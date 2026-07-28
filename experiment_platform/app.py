"""CQ Generation Experiment Platform - Unified Gradio Application.

Supports three CQ generation methods: LLM4KE, Retrofit-CQ, OntologyAgent.
Provides single-run and batch comparison modes with streaming progress.
"""

import glob as glob_mod
import json
import os
import sys
import threading
import time
import traceback

import gradio as gr

# Add experiment_platform to path for imports
_PLATFORM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLATFORM_DIR not in sys.path:
    sys.path.insert(0, _PLATFORM_DIR)

from dataset_registry import get_dataset_names, get_dataset_info, load_ground_truth, load_ground_truth_labels
from evaluation.unified_eval import evaluate
from runners.llm4ke_runner import LLM4KERunner
from runners.retrofit_runner import RetrofitRunner
from runners.ontology_agent_runner import OntologyAgentRunner
from runners.monolithic_runner import MonolithicRunner

# --- Constants ---
AVAILABLE_DATASETS = get_dataset_names()
METHODS = {
    "LLM4KE": LLM4KERunner(),
    "Retrofit-CQ": RetrofitRunner(),
    "OntologyAgent": OntologyAgentRunner(),
    "Monolithic": MonolithicRunner(),
}

PROVIDER_PRESETS = {
    "DashScope (qwen-max)": {
        "api_type": "openai",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    },
    "Azure (gpt-5)": {
        "api_type": "azure",
        "model": "gpt-5",
        "base_url": "https://YOUR_AZURE_ENDPOINT.openai.azure.com/",
        "api_key": os.environ.get("AZURE_OPENAI_KEY", os.environ.get("subscription_key", "")),
        "api_version": "2024-12-01-preview",
    },
    "Ollama (local)": {
        "api_type": "ollama",
        "model": "llama3.3:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    "Custom": {
        "api_type": "openai",
        "model": "",
        "base_url": "",
        "api_key": "",
    },
}

DEFAULT_PROVIDER = "DashScope (qwen-max)"


class RunStateManager:
    """Thread-safe server-side progress storage for surviving page refreshes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runs = {}  # key: "single" or "batch"

    def start(self, run_type):
        with self._lock:
            self._runs[run_type] = {
                "status": "running",
                "progress_log": "",
                "outputs": {},
                "started_at": time.time(),
            }

    def append_log(self, run_type, msg):
        with self._lock:
            state = self._runs.get(run_type)
            if state:
                state["progress_log"] += msg + "\n"

    def set_log(self, run_type, log_text):
        with self._lock:
            state = self._runs.get(run_type)
            if state:
                state["progress_log"] = log_text

    def finish(self, run_type, outputs):
        with self._lock:
            state = self._runs.get(run_type)
            if state:
                state["status"] = "completed"
                state["outputs"] = outputs

    def error(self, run_type):
        with self._lock:
            state = self._runs.get(run_type)
            if state:
                state["status"] = "error"

    def get_state(self, run_type):
        with self._lock:
            state = self._runs.get(run_type)
            if state is None:
                return None
            return dict(state)  # shallow copy

    def clear(self, run_type):
        with self._lock:
            self._runs.pop(run_type, None)


state_manager = RunStateManager()


def on_provider_change(provider):
    """Auto-fill model/key/url/api_version when provider changes."""
    preset = PROVIDER_PRESETS.get(provider, {})
    return (
        preset.get("model", ""),
        preset.get("api_key", ""),
        preset.get("base_url", ""),
        preset.get("api_version", ""),
    )


def on_method_change(method_name):
    """Update method-specific params visibility."""
    is_llm4ke = method_name == "LLM4KE"
    is_retrofit = method_name == "Retrofit-CQ"
    is_oa = method_name == "OntologyAgent"
    is_mono = method_name == "Monolithic"
    return (
        gr.update(visible=is_llm4ke),
        gr.update(visible=is_retrofit),
        gr.update(visible=is_oa),
        gr.update(visible=is_mono),
    )


def _build_llm_config(provider, model, api_key, base_url, api_version=""):
    """Build LLM config dict from UI inputs."""
    preset = PROVIDER_PRESETS.get(provider, {})
    config = {
        "api_type": preset.get("api_type", "openai"),
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }
    if api_version:
        config["api_version"] = api_version
    return config


def _get_method_params(method_name, llm4ke_task, llm4ke_n_cqs, llm4ke_n_examples,
                       retrofit_deeponto, retrofit_concurrency,
                       oa_segmentation, oa_max_iter,
                       oa_cq_examples, oa_max_sub_triples,
                       oa_skip_segmentation, oa_skip_validator,
                       mono_use_chunking, mono_chunk_size, mono_cq_examples):
    """Build method-specific params dict."""
    if method_name == "LLM4KE":
        return {
            "task": llm4ke_task,
            "n_cqs": int(llm4ke_n_cqs),
            "n_examples": int(llm4ke_n_examples),
        }
    elif method_name == "Retrofit-CQ":
        return {
            "use_deeponto": retrofit_deeponto,
            "concurrency": int(retrofit_concurrency),
        }
    elif method_name == "OntologyAgent":
        return {
            "segmentation_method": oa_segmentation,
            "max_iterations": int(oa_max_iter),
            "cq_examples_num": int(oa_cq_examples),
            "max_sub_triples": int(oa_max_sub_triples),
            "skip_segmentation": oa_skip_segmentation,
            "skip_validator": oa_skip_validator,
        }
    elif method_name == "Monolithic":
        return {
            "use_chunking": mono_use_chunking,
            "chunk_size": int(mono_chunk_size),
            "cq_examples_num": int(mono_cq_examples),
        }
    return {}


def _save_run_result(run_result, eval_metrics, llm_config=None):
    """Save complete run result + evaluation to experiment_platform/results/."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = (llm_config or {}).get("model", "unknown")
    # Sanitize model name for directory (replace / and other unsafe chars)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    results_dir = os.path.join(
        _PLATFORM_DIR, "results", run_result.dataset, run_result.method, safe_model_name
    )
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "method": run_result.method,
        "dataset": run_result.dataset,
        "model": model_name,
        "api_type": (llm_config or {}).get("api_type", "unknown"),
        "timestamp": timestamp,
        "duration_seconds": round(run_result.duration_seconds, 2),
        "generated_cqs": run_result.generated_cqs,
        "metrics": {
            **run_result.metrics,
            **{k: v for k, v in eval_metrics.items()
               if k not in ("per_cq_scores", "match_details")},
        },
    }

    path = os.path.join(results_dir, f"run_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return path


# --- Single Run ---
def _single_worker(dataset_name, llm_config, method_name, params,
                   eval_threshold, eval_dedup, eval_dedup_threshold):
    """Background worker for single CQ generation run.

    Runs entirely in a background thread so it survives client disconnects.
    All progress is written to state_manager which the polling generator reads.
    """
    runner = METHODS.get(method_name)
    if runner is None:
        state_manager.append_log("single", f"Unknown method: {method_name}")
        state_manager.error("single")
        return

    def progress_callback(msg):
        state_manager.append_log("single", msg)

    try:
        run_result = runner.run(dataset_name, llm_config, params, progress_callback)
    except Exception:
        error = traceback.format_exc()
        state_manager.append_log("single", f"\n[ERROR] {error}")
        state_manager.append_log("single", f"\n\nPipeline failed.")
        state_manager.error("single")
        return

    if run_result is None:
        state_manager.append_log("single", "\n\nNo results returned.")
        state_manager.error("single")
        return

    # Format CQs
    cqs = run_result.generated_cqs
    cqs_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(cqs))

    # Evaluate
    state_manager.append_log("single",
        f"\n\nEvaluating {len(cqs)} generated CQs against ground truth...")

    metrics_text = ""
    metrics = {}
    try:
        gt_cqs = load_ground_truth(dataset_name)
        gt_labels = load_ground_truth_labels(dataset_name)
        metrics = evaluate(cqs, gt_cqs, threshold=eval_threshold,
                           dedup=eval_dedup, dedup_threshold=eval_dedup_threshold,
                           gt_labels=gt_labels)
        run_result.metrics.update(metrics)

        dedup_info = ""
        if eval_dedup and metrics.get("generated_count_before_dedup", 0) != metrics["generated_count"]:
            removed = metrics["generated_count_before_dedup"] - metrics["generated_count"]
            dedup_info = (
                f"\nDedup: {metrics['generated_count_before_dedup']} -> "
                f"{metrics['generated_count']} ({removed} duplicates removed)\n"
            )

        label_info = ""
        if "recall_simple" in metrics:
            label_info = (
                f"\nRecall (Simple):  {metrics['recall_simple']:.4f}"
                f"  ({metrics['simple_matched']}/{metrics['simple_total']})\n"
                f"Recall (Complex): {metrics['recall_complex']:.4f}"
                f"  ({metrics['complex_matched']}/{metrics['complex_total']})\n"
            )

        # Entity coverage info
        coverage_info = ""
        coverage_stats = run_result.metrics.get("coverage_stats", {})
        if coverage_stats:
            rate = coverage_stats.get("overall_coverage_rate",
                                      coverage_stats.get("coverage_rate", 0))
            covered = coverage_stats.get("total_covered_entities",
                                         coverage_stats.get("covered_entities", 0))
            total = coverage_stats.get("total_entities", 0)
            coverage_info = f"\nEntity Coverage:  {rate:.2f}%  ({covered}/{total})\n"

            # Show iteration history if available
            iter_history = run_result.metrics.get("iteration_coverage_history", [])
            if iter_history and len(iter_history) > 1:
                coverage_info += "  Per-iteration avg coverage:\n"
                for ih in iter_history:
                    coverage_info += f"    Iter {ih['iteration']+1}: {ih['avg_coverage_rate']:.2f}%\n"

        metrics_text = (
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall:    {metrics['recall']:.4f}\n"
            f"F1 Score:  {metrics['f1']:.4f}\n"
            f"\n"
            f"Generated CQs:    {metrics['generated_count']}\n"
            f"Ground Truth CQs: {metrics['ground_truth_count']}\n"
            f"Matched:          {metrics['matched_count']}\n"
            f"Duration:         {run_result.duration_seconds:.1f}s\n"
            f"{dedup_info}"
            f"{label_info}"
            f"{coverage_info}"
        )
        # Persist result to disk
        saved_path = _save_run_result(run_result, metrics, llm_config)
        state_manager.append_log("single", f"Results saved to {saved_path}")
    except Exception as e:
        metrics_text = f"Evaluation error: {e}"
        metrics = {}

    # Details JSON
    details = {
        "method": run_result.method,
        "dataset": run_result.dataset,
        "duration_seconds": round(run_result.duration_seconds, 2),
        "generated_cqs_count": len(cqs),
        "metrics": {k: v for k, v in metrics.items() if k != "per_cq_scores" and k != "match_details"},
    }
    details_json = json.dumps(details, ensure_ascii=False, indent=2)

    state_manager.append_log("single", "Done!")
    state_manager.finish("single", {
        "cqs": cqs_text,
        "metrics": metrics_text,
        "details": details_json,
    })


def run_single(provider, model, api_key, base_url, api_version, dataset_name, method_name,
               eval_threshold, eval_dedup, eval_dedup_threshold,
               llm4ke_task, llm4ke_n_cqs, llm4ke_n_examples,
               retrofit_deeponto, retrofit_concurrency,
               oa_segmentation, oa_max_iter,
               oa_cq_examples, oa_max_sub_triples,
               oa_skip_segmentation, oa_skip_validator,
               mono_use_chunking, mono_chunk_size, mono_cq_examples):
    """Generator that yields (progress_log, cqs_text, metrics_text, details_json).

    All computation runs in a background worker thread. This generator only
    polls state_manager for progress, so client disconnects do not interrupt
    the actual work.
    """
    state_manager.start("single")

    llm_config = _build_llm_config(provider, model, api_key, base_url, api_version)
    params = _get_method_params(
        method_name, llm4ke_task, llm4ke_n_cqs, llm4ke_n_examples,
        retrofit_deeponto, retrofit_concurrency,
        oa_segmentation, oa_max_iter,
        oa_cq_examples, oa_max_sub_triples,
        oa_skip_segmentation, oa_skip_validator,
        mono_use_chunking, mono_chunk_size, mono_cq_examples,
    )

    runner = METHODS.get(method_name)
    if runner is None:
        state_manager.error("single")
        yield f"Unknown method: {method_name}", "", "", "{}"
        return

    # Start background worker thread — survives client disconnects
    thread = threading.Thread(
        target=_single_worker,
        args=(dataset_name, llm_config, method_name, params,
              eval_threshold, eval_dedup, eval_dedup_threshold),
        daemon=True,
    )
    thread.start()

    # Poll state_manager for progress
    prev_log = ""
    while True:
        time.sleep(0.5)
        state = state_manager.get_state("single")
        if state is None:
            break

        log = state["progress_log"]
        outputs = state.get("outputs", {})

        if state["status"] in ("completed", "error"):
            yield (log, outputs.get("cqs", ""), outputs.get("metrics", ""),
                   outputs.get("details", "{}"))
            return

        if log != prev_log:
            prev_log = log
            yield log, "", "", "{}"


# --- Batch Comparison ---
def _batch_worker(selected_methods, dataset_name, llm_config,
                  eval_threshold, eval_dedup, eval_dedup_threshold,
                  method_params_map):
    """Background worker for batch comparison.

    Runs all methods sequentially, evaluates, and saves results — entirely
    in a background thread so it survives client disconnects / page refreshes.
    """
    # Load ground truth once
    try:
        gt_cqs = load_ground_truth(dataset_name)
        gt_labels = load_ground_truth_labels(dataset_name)
        state_manager.append_log("batch",
            f"Loaded {len(gt_cqs)} ground truth CQs for {dataset_name}\n")
    except Exception as e:
        state_manager.append_log("batch", f"Failed to load ground truth: {e}")
        state_manager.error("batch")
        return

    all_results = []

    for method_name in selected_methods:
        runner = METHODS.get(method_name)
        if runner is None:
            state_manager.append_log("batch", f"Unknown method: {method_name}, skipping")
            continue

        state_manager.append_log("batch",
            f"\n{'='*60}\nRunning {method_name}...\n{'='*60}")

        params = method_params_map.get(method_name, {})

        def progress_callback(msg):
            state_manager.append_log("batch", msg)

        try:
            run_result = runner.run(dataset_name, llm_config, params, progress_callback)
        except Exception:
            error = traceback.format_exc()
            state_manager.append_log("batch", f"\n[ERROR] {error}")
            state_manager.append_log("batch", f"\n{method_name} failed.")
            all_results.append({
                "method": method_name, "status": "FAILED", "generated_cqs": 0,
                "precision": "-", "recall": "-", "f1": "-",
                "recall_simple": "-", "recall_complex": "-",
                "coverage": "-", "duration": "-",
            })
            continue

        if run_result is None:
            all_results.append({
                "method": method_name, "status": "NO RESULT", "generated_cqs": 0,
                "precision": "-", "recall": "-", "f1": "-",
                "recall_simple": "-", "recall_complex": "-",
                "coverage": "-", "duration": "-",
            })
            continue

        # Evaluate
        state_manager.append_log("batch",
            f"\nEvaluating {method_name} ({len(run_result.generated_cqs)} CQs)...")

        try:
            metrics = evaluate(run_result.generated_cqs, gt_cqs, threshold=eval_threshold,
                               dedup=eval_dedup, dedup_threshold=eval_dedup_threshold,
                               gt_labels=gt_labels)

            # Extract coverage rate from runner metrics
            cov_stats = run_result.metrics.get("coverage_stats", {})
            cov_rate = cov_stats.get("overall_coverage_rate",
                                     cov_stats.get("coverage_rate", None))
            cov_str = f"{cov_rate:.2f}%" if cov_rate is not None else "-"

            row = {
                "method": method_name, "status": "OK",
                "generated_cqs": metrics["generated_count"],
                "precision": f"{metrics['precision']:.4f}",
                "recall": f"{metrics['recall']:.4f}",
                "f1": f"{metrics['f1']:.4f}",
                "recall_simple": f"{metrics['recall_simple']:.4f}" if "recall_simple" in metrics else "-",
                "recall_complex": f"{metrics['recall_complex']:.4f}" if "recall_complex" in metrics else "-",
                "coverage": cov_str,
                "duration": f"{run_result.duration_seconds:.1f}s",
            }
            all_results.append(row)

            # Persist result to disk
            saved_path = _save_run_result(run_result, metrics, llm_config)
            state_manager.append_log("batch", f"Results saved to {saved_path}")

            log_line = (
                f"{method_name}: P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
            )
            if "recall_simple" in metrics:
                log_line += (
                    f" R(S)={metrics['recall_simple']:.4f}"
                    f" R(C)={metrics['recall_complex']:.4f}"
                )
            state_manager.append_log("batch", log_line + "\n")
        except Exception as e:
            state_manager.append_log("batch", f"Evaluation error for {method_name}: {e}")
            all_results.append({
                "method": method_name, "status": "EVAL ERROR",
                "generated_cqs": len(run_result.generated_cqs),
                "precision": "-", "recall": "-", "f1": "-",
                "recall_simple": "-", "recall_complex": "-",
                "coverage": "-",
                "duration": f"{run_result.duration_seconds:.1f}s",
            })

    # Build comparison table
    table_rows = []
    for r in all_results:
        table_rows.append([
            r["method"], r["status"], r["generated_cqs"],
            r["precision"], r["recall"], r["f1"],
            r["recall_simple"], r["recall_complex"],
            r["coverage"], r["duration"],
        ])

    details_json = json.dumps(all_results, ensure_ascii=False, indent=2)
    state_manager.append_log("batch", "\n" + "=" * 60 + "\nBatch comparison complete!")
    state_manager.finish("batch", {
        "table": table_rows,
        "details": details_json,
    })


def run_batch(provider, model, api_key, base_url, api_version, dataset_name,
              selected_methods, eval_threshold, eval_dedup, eval_dedup_threshold,
              llm4ke_task, llm4ke_n_cqs, llm4ke_n_examples,
              retrofit_deeponto, retrofit_concurrency,
              oa_segmentation, oa_max_iter,
              oa_cq_examples, oa_max_sub_triples,
              oa_skip_segmentation, oa_skip_validator,
              mono_use_chunking, mono_chunk_size, mono_cq_examples):
    """Generator that yields (progress_log, comparison_table, details_json).

    All computation runs in a background worker thread. This generator only
    polls state_manager for progress, so client disconnects do not interrupt
    the actual work (methods continue running, results get saved).
    """
    if not selected_methods:
        yield "No methods selected.", [], "{}"
        return

    state_manager.start("batch")
    llm_config = _build_llm_config(provider, model, api_key, base_url, api_version)

    # Build method params for all selected methods
    method_params_map = {}
    for m in selected_methods:
        method_params_map[m] = _get_method_params(
            m, llm4ke_task, llm4ke_n_cqs, llm4ke_n_examples,
            retrofit_deeponto, retrofit_concurrency,
            oa_segmentation, oa_max_iter,
            oa_cq_examples, oa_max_sub_triples,
            oa_skip_segmentation, oa_skip_validator,
            mono_use_chunking, mono_chunk_size, mono_cq_examples,
        )

    # Start background worker thread — survives client disconnects
    thread = threading.Thread(
        target=_batch_worker,
        args=(selected_methods, dataset_name, llm_config, eval_threshold,
              eval_dedup, eval_dedup_threshold, method_params_map),
        daemon=True,
    )
    thread.start()

    # Poll state_manager for progress
    prev_log = ""
    while True:
        time.sleep(0.5)
        state = state_manager.get_state("batch")
        if state is None:
            break

        log = state["progress_log"]
        outputs = state.get("outputs", {})

        if state["status"] in ("completed", "error"):
            yield log, outputs.get("table", []), outputs.get("details", "{}")
            return

        if log != prev_log:
            prev_log = log
            yield log, [], "{}"


# --- Results Browser helpers ---
_RESULTS_DIR = os.path.join(_PLATFORM_DIR, "results")

# Cache: list of parsed result records (refreshed on demand)
_results_cache = {"records": [], "mtime": 0}


def _scan_results(force=False):
    """Scan results directory and return list of result records.

    Each record is a dict with keys from the JSON plus 'file_path'.
    Uses a simple mtime-based cache to avoid re-scanning on every call.
    """
    try:
        cur_mtime = os.path.getmtime(_RESULTS_DIR)
    except OSError:
        return []

    if not force and _results_cache["records"] and _results_cache["mtime"] >= cur_mtime:
        return _results_cache["records"]

    records = []
    pattern = os.path.join(_RESULTS_DIR, "*", "*", "*", "run_*.json")
    for path in sorted(glob_mod.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["file_path"] = path
            records.append(data)
        except Exception:
            continue

    _results_cache["records"] = records
    _results_cache["mtime"] = cur_mtime
    return records


def _build_results_table(dataset_filter="All"):
    """Build comparison table from saved results.

    Returns (table_rows, run_index_list).
    run_index_list maps row index -> record for CQ detail lookup.
    """
    records = _scan_results(force=True)

    if dataset_filter and dataset_filter != "All":
        records = [r for r in records if r.get("dataset") == dataset_filter]

    table_rows = []
    for r in records:
        m = r.get("metrics", {})
        cov_stats = m.get("coverage_stats", {})
        cov_rate = cov_stats.get("overall_coverage_rate",
                                 cov_stats.get("coverage_rate", None))
        cov_str = f"{cov_rate:.2f}%" if cov_rate is not None else "-"

        table_rows.append([
            r.get("dataset", "?"),
            r.get("method", "?"),
            r.get("model", "?"),
            m.get("generated_count", len(r.get("generated_cqs", []))),
            f"{m['precision']:.4f}" if "precision" in m else "-",
            f"{m['recall']:.4f}" if "recall" in m else "-",
            f"{m['f1']:.4f}" if "f1" in m else "-",
            f"{m['recall_simple']:.4f}" if "recall_simple" in m else "-",
            f"{m['recall_complex']:.4f}" if "recall_complex" in m else "-",
            cov_str,
            f"{r.get('duration_seconds', 0):.1f}s",
            r.get("timestamp", "?"),
        ])

    return table_rows


def _get_result_datasets():
    """Get list of datasets that have saved results."""
    records = _scan_results(force=True)
    datasets = sorted(set(r.get("dataset", "?") for r in records))
    return ["All"] + datasets


def _on_results_refresh(dataset_filter):
    """Refresh results table and dataset dropdown."""
    datasets = _get_result_datasets()
    if dataset_filter not in datasets:
        dataset_filter = "All"
    table = _build_results_table(dataset_filter)
    return gr.update(choices=datasets, value=dataset_filter), table, "", ""


def _on_results_dataset_change(dataset_filter):
    """Filter results table by dataset."""
    table = _build_results_table(dataset_filter)
    return table, "", ""


def _on_results_row_select(evt: gr.SelectData, dataset_filter):
    """Show generated CQs and metrics for the selected row."""
    records = _scan_results()
    if dataset_filter and dataset_filter != "All":
        records = [r for r in records if r.get("dataset") == dataset_filter]

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx < 0 or row_idx >= len(records):
        return "Invalid selection", ""

    r = records[row_idx]
    cqs = r.get("generated_cqs", [])
    cqs_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(cqs))

    m = r.get("metrics", {})
    cov_stats = m.get("coverage_stats", {})
    cov_rate = cov_stats.get("overall_coverage_rate",
                             cov_stats.get("coverage_rate", None))

    dedup_info = ""
    if m.get("generated_count_before_dedup", 0) and m.get("generated_count_before_dedup") != m.get("generated_count"):
        removed = m["generated_count_before_dedup"] - m["generated_count"]
        dedup_info = (
            f"\nDedup: {m['generated_count_before_dedup']} -> "
            f"{m['generated_count']} ({removed} removed)\n"
        )

    label_info = ""
    if "recall_simple" in m:
        label_info = (
            f"\nRecall (Simple):  {m['recall_simple']:.4f}"
            f"  ({m.get('simple_matched', '?')}/{m.get('simple_total', '?')})\n"
            f"Recall (Complex): {m['recall_complex']:.4f}"
            f"  ({m.get('complex_matched', '?')}/{m.get('complex_total', '?')})\n"
        )

    coverage_info = ""
    if cov_rate is not None:
        covered = cov_stats.get("total_covered_entities",
                                cov_stats.get("covered_entities", 0))
        total = cov_stats.get("total_entities", 0)
        coverage_info = f"\nEntity Coverage:  {cov_rate:.2f}%  ({covered}/{total})\n"

    metrics_text = (
        f"Dataset:  {r.get('dataset', '?')}\n"
        f"Method:   {r.get('method', '?')}\n"
        f"Model:    {r.get('model', '?')}\n"
        f"Time:     {r.get('timestamp', '?')}\n"
        f"Duration: {r.get('duration_seconds', 0):.1f}s\n"
        f"\n"
        f"Precision: {m.get('precision', 0):.4f}\n"
        f"Recall:    {m.get('recall', 0):.4f}\n"
        f"F1 Score:  {m.get('f1', 0):.4f}\n"
        f"\n"
        f"Generated CQs:    {m.get('generated_count', len(cqs))}\n"
        f"Ground Truth CQs: {m.get('ground_truth_count', '?')}\n"
        f"Matched:          {m.get('matched_count', '?')}\n"
        f"{dedup_info}"
        f"{label_info}"
        f"{coverage_info}"
    )

    return cqs_text, metrics_text


# --- UI ---
with gr.Blocks(title="CQ Generation Experiment Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CQ Generation Experiment Platform")
    gr.Markdown(
        "Compare CQ generation methods (LLM4KE, Retrofit-CQ, OntologyAgent, Monolithic) "
        "on ontology datasets with unified SBERT evaluation. "
        "Supports ablation experiments for OntologyAgent (w/o Segmenter, w/o Validator)."
    )

    with gr.Row():
        # ---- Left panel: Common Config ----
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### LLM Configuration")
            provider_dropdown = gr.Dropdown(
                choices=list(PROVIDER_PRESETS.keys()),
                value=DEFAULT_PROVIDER,
                label="Provider Preset",
            )
            model_textbox = gr.Textbox(
                value=PROVIDER_PRESETS[DEFAULT_PROVIDER]["model"],
                label="Model Name",
            )
            api_key_textbox = gr.Textbox(
                value=PROVIDER_PRESETS[DEFAULT_PROVIDER]["api_key"],
                label="API Key",
                type="password",
            )
            base_url_textbox = gr.Textbox(
                value=PROVIDER_PRESETS[DEFAULT_PROVIDER]["base_url"],
                label="Base URL",
            )
            api_version_textbox = gr.Textbox(
                value=PROVIDER_PRESETS[DEFAULT_PROVIDER].get("api_version", ""),
                label="API Version (Azure only)",
            )

            gr.Markdown("### Dataset")
            dataset_dropdown = gr.Dropdown(
                choices=AVAILABLE_DATASETS,
                value=AVAILABLE_DATASETS[0] if AVAILABLE_DATASETS else None,
                label="Dataset",
            )
            eval_threshold = gr.Slider(
                minimum=0.3, maximum=0.9, value=0.6, step=0.05,
                label="Eval Similarity Threshold",
            )
            eval_dedup = gr.Checkbox(
                value=True, label="Deduplicate CQs before evaluation",
            )
            eval_dedup_threshold = gr.Slider(
                minimum=0.7, maximum=0.95, value=0.85, step=0.05,
                label="Dedup Similarity Threshold",
            )

            # --- Method-specific params (shown/hidden based on method selection) ---
            gr.Markdown("### Method Parameters")

            with gr.Group(visible=True) as llm4ke_params_group:
                gr.Markdown("**LLM4KE**")
                llm4ke_task = gr.Dropdown(
                    choices=["all_classes", "all_classes+properties", "logic"],
                    value="all_classes",
                    label="Task Type",
                )
                llm4ke_n_cqs = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="CQs per Batch",
                )
                llm4ke_n_examples = gr.Slider(
                    minimum=0, maximum=10, value=0, step=1,
                    label="Few-shot Examples",
                )

            with gr.Group(visible=True) as retrofit_params_group:
                gr.Markdown("**Retrofit-CQ**")
                retrofit_deeponto = gr.Checkbox(
                    value=False, label="Use DeepOnto (instead of rdflib)",
                )
                retrofit_concurrency = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Concurrent Requests",
                )

            with gr.Group(visible=True) as oa_params_group:
                gr.Markdown("**OntologyAgent**")
                oa_segmentation = gr.Dropdown(
                    choices=["auto", "metis", "louvain", "leiden", "spectral", "random"],
                    value="auto",
                    label="Segmentation Method",
                )
                oa_max_iter = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Max Iterations",
                )
                oa_cq_examples = gr.Slider(
                    minimum=1, maximum=30, value=5, step=1,
                    label="CQ Examples Count",
                )
                oa_max_sub_triples = gr.Slider(
                    minimum=10, maximum=100, value=25, step=5,
                    label="Max Triples per Subgraph",
                )
                gr.Markdown("*Ablation Controls*")
                oa_skip_segmentation = gr.Checkbox(
                    value=False,
                    label="Skip Segmentation (w/o Segmenter)",
                )
                oa_skip_validator = gr.Checkbox(
                    value=False,
                    label="Skip SPARQL Validator (w/o Validator)",
                )

            with gr.Group(visible=True) as mono_params_group:
                gr.Markdown("**Monolithic**")
                mono_use_chunking = gr.Checkbox(
                    value=False,
                    label="Use Chunking (split triples into batches)",
                )
                mono_chunk_size = gr.Slider(
                    minimum=10, maximum=100, value=25, step=5,
                    label="Chunk Size (triples per chunk)",
                )
                mono_cq_examples = gr.Slider(
                    minimum=0, maximum=30, value=10, step=1,
                    label="Few-shot Examples Count",
                )

        # ---- Right panel: Tabs ----
        with gr.Column(scale=2):
            with gr.Tabs():
                # --- Tab: Single Run ---
                with gr.Tab("Single Run"):
                    method_radio = gr.Radio(
                        choices=list(METHODS.keys()),
                        value="LLM4KE",
                        label="Method",
                    )
                    single_run_btn = gr.Button(
                        "Run", variant="primary", size="lg",
                    )
                    single_progress = gr.Textbox(
                        label="Progress Log",
                        lines=15, max_lines=30,
                        interactive=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            single_cqs = gr.Textbox(
                                label="Generated CQs",
                                lines=12, max_lines=25,
                                interactive=False,
                            )
                        with gr.Column():
                            single_metrics = gr.Textbox(
                                label="Metrics (P/R/F1)",
                                lines=8, max_lines=12,
                                interactive=False,
                            )
                    single_details = gr.Textbox(
                        label="Details (JSON)",
                        lines=6, max_lines=15,
                        interactive=False,
                    )
                    single_timer = gr.Timer(value=1, active=False)

                # --- Tab: Batch Comparison ---
                with gr.Tab("Batch Comparison"):
                    methods_checkbox = gr.CheckboxGroup(
                        choices=list(METHODS.keys()),
                        value=list(METHODS.keys()),
                        label="Methods to Compare",
                    )
                    batch_run_btn = gr.Button(
                        "Run All", variant="primary", size="lg",
                    )
                    batch_progress = gr.Textbox(
                        label="Progress Log",
                        lines=15, max_lines=30,
                        interactive=False,
                    )
                    batch_table = gr.Dataframe(
                        headers=[
                            "Method", "Status", "Generated CQs",
                            "Precision", "Recall", "F1",
                            "R(Simple)", "R(Complex)", "Coverage", "Duration",
                        ],
                        label="Comparison Results",
                        interactive=False,
                    )
                    batch_details = gr.Textbox(
                        label="Details (JSON)",
                        lines=8, max_lines=15,
                        interactive=False,
                    )
                    batch_timer = gr.Timer(value=1, active=False)

                # --- Tab: Results Browser ---
                with gr.Tab("Results Browser"):
                    with gr.Row():
                        results_dataset_filter = gr.Dropdown(
                            choices=_get_result_datasets(),
                            value="All",
                            label="Filter by Dataset",
                            scale=2,
                        )
                        results_refresh_btn = gr.Button(
                            "Refresh", variant="secondary", scale=1,
                        )
                    results_table = gr.Dataframe(
                        headers=[
                            "Dataset", "Method", "Model", "Generated CQs",
                            "Precision", "Recall", "F1",
                            "R(Simple)", "R(Complex)", "Coverage",
                            "Duration", "Timestamp",
                        ],
                        value=_build_results_table("All"),
                        label="Saved Results (click a row to view details)",
                        interactive=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            results_cqs = gr.Textbox(
                                label="Generated CQs",
                                lines=15, max_lines=30,
                                interactive=False,
                            )
                        with gr.Column():
                            results_metrics = gr.Textbox(
                                label="Run Details",
                                lines=15, max_lines=30,
                                interactive=False,
                            )

                # --- Tab: Ablation Experiments ---
                with gr.Tab("Ablation Experiments"):
                    gr.Markdown(
                        "Run ablation experiments and generate paper tables. "
                        "API keys are read from `llm_configs.json` or environment variables "
                        "(`DASHSCOPE_API_KEY`, `AZURE_OPENAI_KEY`)."
                    )
                    with gr.Row():
                        abl_tables = gr.CheckboxGroup(
                            choices=["A", "B", "C"],
                            value=["A", "B", "C"],
                            label="Tables (A: MAS vs Monolithic, B: Segmentation, C: Component Ablation)",
                        )
                    with gr.Row():
                        abl_dataset = gr.Dropdown(
                            choices=["All", "onem2m", "saref4env", "videogameontology", "vicinitycore"],
                            value="All",
                            label="Dataset Filter",
                        )
                        abl_model = gr.Dropdown(
                            choices=["All", "qwen-max", "glm-5", "gpt-5"],
                            value="All",
                            label="Model Filter",
                        )
                        abl_dry_run = gr.Checkbox(value=False, label="Dry Run (preview only)")
                        abl_force = gr.Checkbox(value=False, label="Force Re-run")
                    with gr.Row():
                        abl_run_btn = gr.Button("Run Ablation", variant="primary", size="lg")
                        abl_refresh_btn = gr.Button("Refresh Tables (from existing results)", variant="secondary")
                    abl_progress = gr.Textbox(
                        label="Progress Log",
                        lines=15, max_lines=30,
                        interactive=False,
                    )
                    gr.Markdown("### Generated Tables")
                    abl_table_a = gr.Textbox(
                        label="Table A: MAS vs Monolithic",
                        lines=6, max_lines=10,
                        interactive=False,
                    )
                    abl_table_b = gr.Textbox(
                        label="Table B: Segmentation Algorithms",
                        lines=10, max_lines=15,
                        interactive=False,
                    )
                    abl_table_c = gr.Textbox(
                        label="Table C: Component Ablation",
                        lines=8, max_lines=12,
                        interactive=False,
                    )
                    abl_timer = gr.Timer(value=1, active=False)

                # --- Tab: Paper Tables ---
                with gr.Tab("Paper Tables"):
                    gr.Markdown(
                        "One-click view of all paper tables (Main + Ablation A/B/C) "
                        "from existing results. Also exports to Excel."
                    )
                    with gr.Row():
                        paper_refresh_btn = gr.Button(
                            "Generate All Tables", variant="primary", size="lg",
                        )
                        paper_xlsx_path = gr.Textbox(
                            label="Excel File", interactive=False, scale=2,
                        )
                    paper_main = gr.Textbox(
                        label="Main Table: Method Comparison (avg across 3 LLMs)",
                        lines=7, max_lines=10, interactive=False,
                    )
                    paper_table_a = gr.Textbox(
                        label="Table A: MAS vs Monolithic (avg across 3 LLMs)",
                        lines=6, max_lines=10, interactive=False,
                    )
                    paper_table_b = gr.Textbox(
                        label="Table B: Segmentation Algorithms (qwen-max)",
                        lines=10, max_lines=15, interactive=False,
                    )
                    paper_table_c = gr.Textbox(
                        label="Table C: Component Ablation (avg across 3 LLMs)",
                        lines=8, max_lines=12, interactive=False,
                    )

    # --- Results Browser Events ---
    results_refresh_btn.click(
        fn=_on_results_refresh,
        inputs=[results_dataset_filter],
        outputs=[results_dataset_filter, results_table, results_cqs, results_metrics],
    )
    results_dataset_filter.change(
        fn=_on_results_dataset_change,
        inputs=[results_dataset_filter],
        outputs=[results_table, results_cqs, results_metrics],
    )
    results_table.select(
        fn=_on_results_row_select,
        inputs=[results_dataset_filter],
        outputs=[results_cqs, results_metrics],
    )

    # --- Polling functions for progress recovery ---
    def poll_single_progress():
        state = state_manager.get_state("single")
        if state is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        outputs = state.get("outputs", {})
        if state["status"] == "running":
            return (
                state["progress_log"],
                outputs.get("cqs", gr.skip()),
                outputs.get("metrics", gr.skip()),
                outputs.get("details", gr.skip()),
                gr.Timer(active=True),
            )
        # completed or error — return final state and stop timer
        return (
            state["progress_log"],
            outputs.get("cqs", ""),
            outputs.get("metrics", ""),
            outputs.get("details", "{}"),
            gr.Timer(active=False),
        )

    def poll_batch_progress():
        state = state_manager.get_state("batch")
        if state is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        outputs = state.get("outputs", {})
        if state["status"] == "running":
            return (
                state["progress_log"],
                outputs.get("table", gr.skip()),
                outputs.get("details", gr.skip()),
                gr.Timer(active=True),
            )
        return (
            state["progress_log"],
            outputs.get("table", []),
            outputs.get("details", "{}"),
            gr.Timer(active=False),
        )

    # --- Ablation experiment helpers ---
    def _ablation_worker(tables, dry_run, force, dataset, model):
        """Background worker for ablation experiments."""
        try:
            from ablation_runner import set_log_callback, run_ablation as _run_abl

            def cb(msg):
                state_manager.append_log("ablation", msg)

            set_log_callback(cb)
            result = _run_abl(tables, dry_run, force, dataset, model)

            state_manager.finish("ablation", {
                "table_a": result.get("tables", {}).get("A", ""),
                "table_b": result.get("tables", {}).get("B", ""),
                "table_c": result.get("tables", {}).get("C", ""),
            })
        except Exception:
            state_manager.append_log("ablation", f"\n[ERROR] {traceback.format_exc()}")
            state_manager.error("ablation")
        finally:
            from ablation_runner import set_log_callback as _set_cb
            _set_cb(None)

    def run_ablation_ui(tables, dataset, model, dry_run, force):
        """Generator that yields (progress, table_a, table_b, table_c)."""
        if not tables:
            yield "No tables selected.", "", "", ""
            return

        state_manager.start("ablation")

        ds = None if dataset == "All" else dataset
        mdl = None if model == "All" else model

        worker = threading.Thread(
            target=_ablation_worker,
            args=(tables, dry_run, force, ds, mdl),
            daemon=True,
        )
        worker.start()

        prev_log = ""
        while True:
            time.sleep(0.5)
            state = state_manager.get_state("ablation")
            if state is None:
                break

            log_text = state["progress_log"]
            outputs = state.get("outputs", {})

            if state["status"] in ("completed", "error"):
                yield (
                    log_text,
                    outputs.get("table_a", ""),
                    outputs.get("table_b", ""),
                    outputs.get("table_c", ""),
                )
                return

            if log_text != prev_log:
                prev_log = log_text
                yield log_text, gr.skip(), gr.skip(), gr.skip()

    def refresh_ablation_tables(tables):
        """Refresh tables from existing results without running experiments."""
        if not tables:
            return "", "", ""
        from ablation_runner import refresh_tables
        result = refresh_tables(tables)
        return (
            result.get("A", "(no data for Table A)"),
            result.get("B", "(no data for Table B)"),
            result.get("C", "(no data for Table C)"),
        )

    def poll_ablation_progress():
        state = state_manager.get_state("ablation")
        if state is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        outputs = state.get("outputs", {})
        if state["status"] == "running":
            return (
                state["progress_log"],
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.Timer(active=True),
            )
        return (
            state["progress_log"],
            outputs.get("table_a", ""),
            outputs.get("table_b", ""),
            outputs.get("table_c", ""),
            gr.Timer(active=False),
        )

    def refresh_paper_tables():
        """Generate all paper tables and return (main, A, B, C, xlsx_path)."""
        from ablation_runner import refresh_tables
        result = refresh_tables(["Main", "A", "B", "C"])
        xlsx_path = os.path.join(
            _PLATFORM_DIR, "results", "ablation", "tables", "ablation_tables.xlsx"
        )
        xlsx_display = xlsx_path if os.path.exists(xlsx_path) else ""
        return (
            result.get("Main", "(no data)"),
            result.get("A", "(no data)"),
            result.get("B", "(no data)"),
            result.get("C", "(no data)"),
            xlsx_display,
        )

    def on_page_load():
        """Restore progress state on page refresh."""
        single_state = state_manager.get_state("single")
        batch_state = state_manager.get_state("batch")
        ablation_state = state_manager.get_state("ablation")

        # Single run outputs
        if single_state and single_state["status"] == "running":
            s_log = single_state["progress_log"]
            s_cqs = ""
            s_metrics = ""
            s_details = "{}"
            s_timer = gr.Timer(active=True)
        elif single_state and single_state["status"] in ("completed", "error"):
            s_log = single_state["progress_log"]
            s_out = single_state.get("outputs", {})
            s_cqs = s_out.get("cqs", "")
            s_metrics = s_out.get("metrics", "")
            s_details = s_out.get("details", "{}")
            s_timer = gr.Timer(active=False)
        else:
            s_log = gr.skip()
            s_cqs = gr.skip()
            s_metrics = gr.skip()
            s_details = gr.skip()
            s_timer = gr.skip()

        # Batch run outputs
        if batch_state and batch_state["status"] == "running":
            b_log = batch_state["progress_log"]
            b_table = []
            b_details = "{}"
            b_timer = gr.Timer(active=True)
        elif batch_state and batch_state["status"] in ("completed", "error"):
            b_log = batch_state["progress_log"]
            b_out = batch_state.get("outputs", {})
            b_table = b_out.get("table", [])
            b_details = b_out.get("details", "{}")
            b_timer = gr.Timer(active=False)
        else:
            b_log = gr.skip()
            b_table = gr.skip()
            b_details = gr.skip()
            b_timer = gr.skip()

        # Ablation outputs
        if ablation_state and ablation_state["status"] == "running":
            a_log = ablation_state["progress_log"]
            a_ta = ""
            a_tb = ""
            a_tc = ""
            a_timer = gr.Timer(active=True)
        elif ablation_state and ablation_state["status"] in ("completed", "error"):
            a_log = ablation_state["progress_log"]
            a_out = ablation_state.get("outputs", {})
            a_ta = a_out.get("table_a", "")
            a_tb = a_out.get("table_b", "")
            a_tc = a_out.get("table_c", "")
            a_timer = gr.Timer(active=False)
        else:
            a_log = gr.skip()
            a_ta = gr.skip()
            a_tb = gr.skip()
            a_tc = gr.skip()
            a_timer = gr.skip()

        return (s_log, s_cqs, s_metrics, s_details, s_timer,
                b_log, b_table, b_details, b_timer,
                a_log, a_ta, a_tb, a_tc, a_timer)

    # --- Events ---
    provider_dropdown.change(
        fn=on_provider_change,
        inputs=[provider_dropdown],
        outputs=[model_textbox, api_key_textbox, base_url_textbox, api_version_textbox],
    )

    # Shared inputs for LLM config + all method params
    common_inputs = [
        provider_dropdown, model_textbox, api_key_textbox, base_url_textbox,
        api_version_textbox, dataset_dropdown,
    ]
    method_param_inputs = [
        llm4ke_task, llm4ke_n_cqs, llm4ke_n_examples,
        retrofit_deeponto, retrofit_concurrency,
        oa_segmentation, oa_max_iter,
        oa_cq_examples, oa_max_sub_triples,
        oa_skip_segmentation, oa_skip_validator,
        mono_use_chunking, mono_chunk_size, mono_cq_examples,
    ]

    eval_inputs = [eval_threshold, eval_dedup, eval_dedup_threshold]

    single_run_btn.click(
        fn=run_single,
        inputs=common_inputs + [method_radio] + eval_inputs + method_param_inputs,
        outputs=[single_progress, single_cqs, single_metrics, single_details],
    ).then(
        fn=lambda: gr.Timer(active=False),
        outputs=[single_timer],
    )
    # Activate single timer when run starts
    single_run_btn.click(
        fn=lambda: gr.Timer(active=True),
        outputs=[single_timer],
    )

    batch_run_btn.click(
        fn=run_batch,
        inputs=common_inputs + [methods_checkbox] + eval_inputs + method_param_inputs,
        outputs=[batch_progress, batch_table, batch_details],
    ).then(
        fn=lambda: gr.Timer(active=False),
        outputs=[batch_timer],
    )
    # Activate batch timer when run starts
    batch_run_btn.click(
        fn=lambda: gr.Timer(active=True),
        outputs=[batch_timer],
    )

    # Timer tick events for polling progress
    single_timer.tick(
        fn=poll_single_progress,
        outputs=[single_progress, single_cqs, single_metrics, single_details, single_timer],
    )
    batch_timer.tick(
        fn=poll_batch_progress,
        outputs=[batch_progress, batch_table, batch_details, batch_timer],
    )

    # --- Ablation Events ---
    abl_run_btn.click(
        fn=run_ablation_ui,
        inputs=[abl_tables, abl_dataset, abl_model, abl_dry_run, abl_force],
        outputs=[abl_progress, abl_table_a, abl_table_b, abl_table_c],
    ).then(
        fn=lambda: gr.Timer(active=False),
        outputs=[abl_timer],
    )
    abl_run_btn.click(
        fn=lambda: gr.Timer(active=True),
        outputs=[abl_timer],
    )
    abl_refresh_btn.click(
        fn=refresh_ablation_tables,
        inputs=[abl_tables],
        outputs=[abl_table_a, abl_table_b, abl_table_c],
    )
    abl_timer.tick(
        fn=poll_ablation_progress,
        outputs=[abl_progress, abl_table_a, abl_table_b, abl_table_c, abl_timer],
    )

    # --- Paper Tables Events ---
    paper_refresh_btn.click(
        fn=refresh_paper_tables,
        outputs=[paper_main, paper_table_a, paper_table_b, paper_table_c, paper_xlsx_path],
    )

    # Restore state on page load/refresh
    demo.load(
        fn=on_page_load,
        outputs=[
            single_progress, single_cqs, single_metrics, single_details, single_timer,
            batch_progress, batch_table, batch_details, batch_timer,
            abl_progress, abl_table_a, abl_table_b, abl_table_c, abl_timer,
        ],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)
