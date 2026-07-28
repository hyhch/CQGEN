"""Post-hoc experiment: run the SPARQL Validator on ground-truth CQs.

Reproduces the protocol described in exp_validator_on_gt.md §3:

  For each ground-truth CQ in each ontology (OneM2M / SAREF4ENV / VGO / VC):
    1. Translate to SPARQL via the Generator prompt   (tau)
    2. Execute against the rebuilt rdflib.Graph        -> first_pass?
    3. If failed, apply CORRECT prompt once            (repair)
    4. Execute again                                   -> post_repair_pass?
    5. Classify failure reason: SYNTAX | UNDEFINED_ENTITY | EXEC_ERROR | OTHER

Per-LLM, per-ontology breakdown is saved as JSON; aggregate stats are printed.

USAGE
    # Fill in API keys in experiment_platform/llm_configs.json first.
    cd experiment_platform
    python validator_on_gt.py --model qwen-max
    python validator_on_gt.py --model glm-5
    python validator_on_gt.py --model gpt-5
    # Or all three sequentially:
    python validator_on_gt.py --all

Output: experiment_platform/results/validator_on_gt/<model>/<ontology>.json
        experiment_platform/results/validator_on_gt/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

# --------------------------------------------------------------------------- #
# Workspace bootstrap (mirror runners/ontology_agent_runner.py)
# --------------------------------------------------------------------------- #
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OA_ROOT = os.path.join(_WORKSPACE, "OntologyAgent")

os.environ.setdefault("METAGPT_PROJECT_ROOT", _OA_ROOT)
import metagpt  # noqa: F401  # import BEFORE OA_ROOT goes on sys.path
if _OA_ROOT not in sys.path:
    sys.path.insert(0, _OA_ROOT)

from rdflib import Graph  # noqa: E402

# Reuse the production Validator unchanged.
from roles.sparql_evaluator import SPARQLQuery  # noqa: E402
from roles.ontology_segmenter import LoadOWL    # noqa: E402


# --------------------------------------------------------------------------- #
# GPT-5 / Azure compatibility shim:
#   gpt-5 deployments reject `max_tokens` and a non-default `temperature`.
#   Rewrite metagpt's OpenAILLM._cons_kwargs to use `max_completion_tokens`
#   and drop `temperature` when targeting gpt-5.
# --------------------------------------------------------------------------- #
def _patch_metagpt_for_gpt5() -> None:
    from metagpt.provider.openai_api import OpenAILLM

    _orig = OpenAILLM._cons_kwargs

    def _patched(self, messages, timeout=None, **extra):
        kw = _orig(self, messages, timeout=timeout, **extra)
        model = (self.model or "").lower()
        if "gpt-5" in model or "gpt5" in model:
            if "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            kw.pop("temperature", None)
        return kw

    OpenAILLM._cons_kwargs = _patched

# --------------------------------------------------------------------------- #
# Dataset registry (matches OntologyAgent/dataset/*/*.json layout)
# --------------------------------------------------------------------------- #
DATASETS = {
    "onem2m":            {"dir": "onem2m",            "label": "OneM2M"},
    "saref4env":         {"dir": "saref4env",         "label": "SAREF4ENV"},
    "videogameontology": {"dir": "videogameontology", "label": "VGO"},
    "vicinitycore":      {"dir": "vicinitycore",      "label": "VC"},
}

DATASET_ROOT = os.path.join(_OA_ROOT, "dataset")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "validator_on_gt")


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #
_SYNTAX_HINTS = (
    "expected", "parse", "syntax", "tokenizing", "unexpected",
    "bad sparql", "bad query", "lexical",
)
_UNDEFINED_HINTS = (
    "unknown prefix", "namespace", "undefined", "not bound", "not declared",
)


def classify_failure(query_text: str, error: Exception, namespaces: list,
                     triples: list) -> str:
    """Best-effort failure classification per exp_validator_on_gt.md §3.3."""
    msg = (str(error) or "").lower()

    if any(h in msg for h in _SYNTAX_HINTS):
        return "SYNTAX"
    if any(h in msg for h in _UNDEFINED_HINTS):
        return "UNDEFINED_ENTITY"

    # Heuristic UNDEFINED_ENTITY: parse succeeded but query references a
    # local-name not present in any triple subject/predicate/object.
    qnames = set(re.findall(r"\b[A-Za-z_][\w-]*:[A-Za-z_][\w-]*\b", query_text))
    if qnames:
        ttext = " ".join(f"{s} {p} {o}" for s, p, o in triples)
        for qn in qnames:
            local = qn.split(":", 1)[1]
            if not local:
                continue
            if local.lower() in {"type", "class", "domain", "range",
                                 "subclassof", "subpropertyof"}:
                continue
            if local not in ttext:
                return "UNDEFINED_ENTITY"

    if "error" in msg or "exception" in msg or msg:
        return "EXEC_ERROR"
    return "OTHER"


# --------------------------------------------------------------------------- #
# Core: validate a single GT CQ
# --------------------------------------------------------------------------- #
async def validate_one_cq(cq: str, namespaces: list, triples: list,
                          sparql_action: SPARQLQuery) -> dict:
    """Translate cq -> SPARQL, try once; if failed, apply CORRECT once."""

    # ----- first pass: τ(cq) ----- #
    gen_prompt = sparql_action.GENERATE_PROMPT_TEMPLATE.format(
        question=cq, triples=triples,
    )
    try:
        rsp = await sparql_action._aask(gen_prompt)
    except Exception as e:
        return {
            "cq": cq, "first_pass": False, "post_repair_pass": False,
            "first_failure_reason": "OTHER",
            "first_error": f"LLM error: {e}",
            "first_sparql": None, "repair_sparql": None,
        }
    sparql_q = sparql_action.extract_query(rsp)

    try:
        sparql_action.execute_sparql_query(sparql_q, namespaces, triples)
        return {
            "cq": cq, "first_pass": True, "post_repair_pass": True,
            "first_failure_reason": None, "first_error": None,
            "first_sparql": sparql_q, "repair_sparql": None,
        }
    except Exception as e_first:
        first_reason = classify_failure(sparql_q, e_first, namespaces, triples)
        first_err = str(e_first)

    # ----- one repair round via CORRECT prompt ----- #
    correct_prompt = sparql_action.CORRECT_PROMPT_TEMPLATE.format(
        sparql_query=sparql_q, question=cq, feedback=first_err, triples=triples,
    )
    try:
        rsp2 = await sparql_action._aask(correct_prompt)
    except Exception as e:
        return {
            "cq": cq, "first_pass": False, "post_repair_pass": False,
            "first_failure_reason": first_reason,
            "first_error": first_err,
            "first_sparql": sparql_q,
            "repair_failure_reason": "OTHER",
            "repair_error": f"LLM error: {e}",
            "repair_sparql": None,
        }
    sparql_q2 = sparql_action.extract_query(rsp2)

    try:
        sparql_action.execute_sparql_query(sparql_q2, namespaces, triples)
        return {
            "cq": cq, "first_pass": False, "post_repair_pass": True,
            "first_failure_reason": first_reason,
            "first_error": first_err,
            "first_sparql": sparql_q,
            "repair_failure_reason": None,
            "repair_error": None,
            "repair_sparql": sparql_q2,
        }
    except Exception as e_rep:
        return {
            "cq": cq, "first_pass": False, "post_repair_pass": False,
            "first_failure_reason": first_reason,
            "first_error": first_err,
            "first_sparql": sparql_q,
            "repair_failure_reason": classify_failure(sparql_q2, e_rep,
                                                      namespaces, triples),
            "repair_error": str(e_rep),
            "repair_sparql": sparql_q2,
        }


# --------------------------------------------------------------------------- #
# Per-ontology runner
# --------------------------------------------------------------------------- #
async def run_ontology(name: str, sparql_action: SPARQLQuery) -> dict:
    info = DATASETS[name]
    j_path = os.path.join(DATASET_ROOT, info["dir"], f"{info['dir']}.json")
    with open(j_path, encoding="utf-8") as f:
        meta = json.load(f)
    cqs = meta["competency_questions"]
    owl_path = os.path.join(DATASET_ROOT, info["dir"], meta["file_name"])

    loader = LoadOWL()
    parsed = await loader.run(owl_path)
    ns = parsed["namespaces"]
    triples = parsed["triples"]

    print(f"  [{info['label']}] {len(cqs)} GT CQs, {len(triples)} triples")

    per_cq = []
    t0 = time.time()
    for i, cq in enumerate(cqs, 1):
        res = await validate_one_cq(cq, ns, triples, sparql_action)
        per_cq.append(res)
        if i % 10 == 0 or i == len(cqs):
            fp = sum(1 for r in per_cq if r["first_pass"])
            rp = sum(1 for r in per_cq if r["post_repair_pass"])
            print(f"    {i}/{len(cqs)}  first={fp}  post_repair={rp}  "
                  f"({time.time()-t0:.0f}s)")

    fp = sum(1 for r in per_cq if r["first_pass"])
    rp = sum(1 for r in per_cq if r["post_repair_pass"])
    n = len(per_cq)
    fail_break: dict[str, int] = {}
    for r in per_cq:
        if r["post_repair_pass"]:
            continue
        reason = r.get("repair_failure_reason") or r.get("first_failure_reason") or "OTHER"
        fail_break[reason] = fail_break.get(reason, 0) + 1

    return {
        "ontology": info["label"],
        "n_gt": n,
        "first_pass_count": fp,
        "first_pass_rate": round(100 * fp / n, 2),
        "post_repair_count": rp,
        "post_repair_rate": round(100 * rp / n, 2),
        "failure_breakdown": fail_break,
        "per_cq": per_cq,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _load_llm_config(model_key: str) -> dict:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "llm_configs.json")) as f:
        cfgs = json.load(f)
    if model_key not in cfgs:
        raise SystemExit(f"Unknown model {model_key}; choices: {list(cfgs)}")
    cfg = cfgs[model_key]
    if cfg.get("api_key", "").startswith("YOUR_API_KEY"):
        raise SystemExit(f"Fill in API key for {model_key} in llm_configs.json")
    return cfg


async def run_model(model_key: str) -> dict:
    cfg = _load_llm_config(model_key)

    if "gpt-5" in model_key or "gpt5" in model_key:
        _patch_metagpt_for_gpt5()

    from metagpt.config2 import Config
    metagpt_cfg = Config.from_llm_config({
        "api_type":    cfg["api_type"],
        "model":       cfg["model"],
        "api_key":     cfg["api_key"],
        "base_url":    cfg["base_url"],
        "api_version": cfg.get("api_version"),
    })
    sparql_action = SPARQLQuery()
    sparql_action.config = metagpt_cfg

    print(f"[validator_on_gt] model={model_key}")
    per_ont: list[dict] = []
    for name in DATASETS:
        per_ont.append(await run_ontology(name, sparql_action))

    total_n  = sum(o["n_gt"] for o in per_ont)
    total_fp = sum(o["first_pass_count"] for o in per_ont)
    total_rp = sum(o["post_repair_count"] for o in per_ont)
    overall = {
        "n_gt_total":         total_n,
        "first_pass_rate":    round(100 * total_fp / total_n, 2),
        "post_repair_rate":   round(100 * total_rp / total_n, 2),
        "first_pass_count":   total_fp,
        "post_repair_count":  total_rp,
    }
    summary = {
        "model": model_key,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "overall": overall,
        "per_ontology": [{k: v for k, v in o.items() if k != "per_cq"}
                         for o in per_ont],
    }

    out_dir = os.path.join(RESULTS_ROOT, model_key)
    os.makedirs(out_dir, exist_ok=True)
    for o in per_ont:
        with open(os.path.join(out_dir, f"{o['ontology']}.json"), "w") as f:
            json.dump(o, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[validator_on_gt] {model_key} done: "
          f"first_pass={overall['first_pass_rate']}%  "
          f"post_repair={overall['post_repair_rate']}%")
    return summary


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen-max", "glm-5", "gpt-5"])
    ap.add_argument("--all", action="store_true",
                    help="run all three models sequentially")
    args = ap.parse_args()

    if args.all:
        models = ["qwen-max", "glm-5", "gpt-5"]
    elif args.model:
        models = [args.model]
    else:
        ap.error("specify --model <name> or --all")

    summaries = []
    for m in models:
        summaries.append(await run_model(m))

    # Cross-model average (what the rebuttal will quote).
    if len(summaries) > 1:
        avg_first  = sum(s["overall"]["first_pass_rate"]  for s in summaries) / len(summaries)
        avg_repair = sum(s["overall"]["post_repair_rate"] for s in summaries) / len(summaries)
        agg = {
            "averaged_over_models": [s["model"] for s in summaries],
            "first_pass_rate_mean":  round(avg_first, 2),
            "post_repair_rate_mean": round(avg_repair, 2),
        }
        with open(os.path.join(RESULTS_ROOT, "summary.json"), "w") as f:
            json.dump({"per_model": summaries, "aggregate": agg},
                      f, indent=2, ensure_ascii=False)
        print(f"[validator_on_gt] aggregate: "
              f"first_pass={agg['first_pass_rate_mean']}%  "
              f"post_repair={agg['post_repair_rate_mean']}%")


if __name__ == "__main__":
    asyncio.run(main())
