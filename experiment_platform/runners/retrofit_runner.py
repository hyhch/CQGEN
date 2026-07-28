"""Runner for RETROFIT-CQs CQ generation method."""

import importlib.util
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .base import BaseRunner, RunResult

# Workspace root
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RETROFIT_ROOT = os.path.join(_WORKSPACE, "RETROFIT-CQs")

# System prompt from RETROFIT-CQs/generate_cqs.py
SYSTEM_PROMPT = (
    "As an ontology engineer, Provide competency questions focused on the context "
    "provided; avoid using narrative questions. competency questions are the questions "
    "that outline the scope of an ontology and provide an idea about the knowledge that "
    "needs to be entailed in the ontology.Please use 1. XXXX this format to generate CQ, "
    "and do not contain any other content"
)


def _load_extract_triples():
    """Load extract_triples module from RETROFIT-CQs via importlib.

    Avoids adding RETROFIT-CQs root to sys.path, which would make its
    config.py shadow OntologyAgent's config/ package.
    """
    module_path = os.path.join(_RETROFIT_ROOT, "extract_triples.py")
    spec = importlib.util.spec_from_file_location("retrofit_extract_triples", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class UnifiedLLMClient:
    """OpenAI-compatible LLM client supporting both OpenAI and Azure endpoints."""

    def __init__(self, llm_config):
        api_type = llm_config.get("api_type", "openai")
        self.model = llm_config.get("model", "qwen-max")

        if api_type == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=llm_config.get("api_key", "no-key"),
                azure_endpoint=llm_config.get("base_url", ""),
                api_version=llm_config.get("api_version", "2024-12-01-preview"),
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=llm_config.get("api_key", "no-key"),
                base_url=llm_config.get("base_url", ""),
            )

    def complete(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


class RetrofitRunner(BaseRunner):
    name = "Retrofit-CQ"

    def run(
        self,
        dataset_name: str,
        llm_config: dict,
        params: dict,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        start = time.time()
        logs = []

        def log(msg):
            self._log(msg, progress_callback)
            logs.append(msg)

        from dataset_registry import get_dataset_info
        info = get_dataset_info(dataset_name)
        retrofit_info = info["retrofit"]
        ontology_path = retrofit_info["ontology_path"]

        use_deeponto = params.get("use_deeponto", False)

        log(f"[Retrofit-CQ] Extracting triples from: {os.path.basename(ontology_path)}")

        # Stage 1: Triple extraction (loaded via importlib to avoid sys.path pollution)
        et_mod = _load_extract_triples()

        # Create temp file for triples CSV
        triples_fd, triples_csv = tempfile.mkstemp(suffix=".csv")
        os.close(triples_fd)

        try:
            if use_deeponto:
                et_mod.extract_triples_deeponto(ontology_path, triples_csv)
            else:
                # Detect format from extension
                ext = os.path.splitext(ontology_path)[1].lower()
                fmt_map = {".owl": "xml", ".rdf": "xml", ".ttl": "turtle"}
                fmt = fmt_map.get(ext, "xml")
                num_triples = et_mod.extract_triples_rdflib(ontology_path, triples_csv, fmt)
                log(f"[Retrofit-CQ] Extracted {num_triples} triples")

            # Stage 2: Generate CQs using unified OpenAI-compatible client
            log(f"[Retrofit-CQ] Generating CQs with {llm_config.get('model', 'unknown')}...")
            client = UnifiedLLMClient(llm_config)

            # Read triples
            import pandas as pd
            df = pd.read_csv(
                triples_csv, sep="\t", header=None,
                names=["Subject", "Predicate", "Object"],
            )
            log(f"[Retrofit-CQ] Processing {len(df)} triples...")

            rows = df.values.tolist()
            all_cqs = []           # final ordered CQ list
            results = [None] * len(rows)  # per-triple results, preserving order
            lock = threading.Lock()
            processed_count = [0]   # mutable counter for progress
            concurrency = params.get("concurrency", 5)

            def _process_triple(idx, row):
                user_prompt = f"{','.join(str(v) for v in row)}?"
                try:
                    response = client.complete(SYSTEM_PROMPT, user_prompt)
                    cqs = []
                    for line in response.splitlines():
                        line = line.strip()
                        line = re.sub(r"^\d+\.\s*", "", line)
                        if line and line != "Error generating question":
                            cqs.append(line)
                    results[idx] = cqs
                except Exception as e:
                    results[idx] = []
                    log(f"[Retrofit-CQ] Error on triple {idx + 1}: {e}")

                with lock:
                    processed_count[0] += 1
                    done = processed_count[0]
                    # Compute running total of CQs from completed results
                    total_cqs = sum(len(r) for r in results if r is not None)
                if done % 10 == 0 or done == len(rows):
                    log(f"[Retrofit-CQ] Processed {done}/{len(rows)} triples, {total_cqs} CQs so far")

            log(f"[Retrofit-CQ] Using {concurrency} concurrent requests")
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_process_triple, i, row)
                           for i, row in enumerate(rows)]
                for f in as_completed(futures):
                    f.result()  # propagate unexpected exceptions

            # Flatten in original order
            for cqs in results:
                if cqs:
                    all_cqs.extend(cqs)

            log(f"[Retrofit-CQ] Generated {len(all_cqs)} CQs total")
        finally:
            if os.path.exists(triples_csv):
                os.unlink(triples_csv)

        # Compute entity coverage
        coverage_stats = {}
        try:
            from evaluation.coverage import load_entities_for_dataset, compute_string_coverage
            entities = load_entities_for_dataset(dataset_name)
            if entities:
                coverage_stats = compute_string_coverage(entities, all_cqs)
                log(f"[Retrofit-CQ] Entity coverage: {coverage_stats['coverage_rate']:.2f}% "
                    f"({coverage_stats['covered_entities']}/{coverage_stats['total_entities']})")
        except Exception as e:
            log(f"[Retrofit-CQ] Coverage computation failed: {e}")

        duration = time.time() - start
        return RunResult(
            method=self.name,
            dataset=dataset_name,
            generated_cqs=all_cqs,
            metrics={"coverage_stats": coverage_stats},
            intermediate_logs=logs,
            duration_seconds=duration,
        )
