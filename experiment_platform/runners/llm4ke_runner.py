"""Runner for LLM4KE CQ generation method."""

import importlib.util
import os
import time
from typing import Callable, Optional

from .base import BaseRunner, RunResult

# Workspace root
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LLM4KE_ROOT = os.path.join(_WORKSPACE, "llm4ke")
_LLM4KE_SRC = os.path.join(_LLM4KE_ROOT, "src")


def _load_llm4ke_module(name, filename=None):
    """Load a module from llm4ke/src/ via importlib to avoid sys.path pollution."""
    if filename is None:
        filename = f"{name}.py"
    module_path = os.path.join(_LLM4KE_SRC, filename)
    spec = importlib.util.spec_from_file_location(f"llm4ke_{name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LLM4KERunner(BaseRunner):
    name = "LLM4KE"

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
        llm4ke_info = info["llm4ke"]
        data_dir = llm4ke_info["data_dir"]
        ontology_name = llm4ke_info["ontology_name"]
        input_path = os.path.join(data_dir, ontology_name)

        log(f"[LLM4KE] Loading ontology: {ontology_name}")

        # Load llm4ke modules via importlib (avoids sys.path pollution)
        ontology_mod = _load_llm4ke_module("ontology")
        template_mod = _load_llm4ke_module("LocalTemplate")
        backend_mod = _load_llm4ke_module("llm_backend")

        # Load ontology
        graph = ontology_mod.load_ontology(input_path)
        classes = ontology_mod.extract_classes(graph)
        properties = ontology_mod.extract_properties(graph)
        schema = ontology_mod.extract_schema(graph, properties)

        task = params.get("task", "all_classes")
        n_cqs = params.get("n_cqs", 10)
        n_examples = params.get("n_examples", 0)
        include_description = params.get("include_description", False)

        description = ontology_mod.load_description(input_path) if include_description else ""
        log(f"[LLM4KE] Ontology has {len(classes)} classes, {len(properties)} properties")
        log(f"[LLM4KE] Task: {task}, n_cqs: {n_cqs}, n_examples: {n_examples}")

        # Load prompt template
        template_path = os.path.join(_LLM4KE_SRC, "prompt_templates", f"{task}.yml")
        prompt_template = template_mod.LocalTemplate.load(template_path)

        # Build batches
        classes_batches = [
            [ontology_mod.simplify(c) for c in batch]
            for batch in ontology_mod.select_in_batches(classes)
        ]
        property_batches = [
            ontology_mod.props_for_classes(schema, batch)
            for batch in classes_batches
        ]
        schema_batches = [
            ontology_mod.schema_for_classes(schema, batch)
            for batch in classes_batches
        ]

        # Load examples if requested
        examples = ""
        if n_examples > 0:
            gt_cqs = ontology_mod.load_ground_truth(input_path)
            examples = "For example:\n -" + "\n -".join(gt_cqs[:n_examples])

        # Build input batches
        input_batches = []
        for c_batch, p_batch, s_batch in zip(
            classes_batches, property_batches, schema_batches
        ):
            ont_input = {
                "name": ontology_name,
                "description": description,
                "n": n_cqs,
                "classes": "\n- ".join(c_batch),
                "properties": "\n- ".join(p_batch),
                "schema": "\n- ".join(f"({s}, {p}, {o})" for s, p, o in s_batch),
                "examples": examples,
            }
            input_batches.append({k: ont_input[k] for k in prompt_template.input})

        log(f"[LLM4KE] Sending {len(input_batches)} batches to LLM...")

        # Create LLM via LangChain with provided config
        api_type = llm_config.get("api_type", "openai")
        if api_type == "azure":
            from langchain_openai import AzureChatOpenAI
            llm = AzureChatOpenAI(
                api_key=llm_config.get("api_key", ""),
                azure_endpoint=llm_config.get("base_url", ""),
                api_version=llm_config.get("api_version", "2024-12-01-preview"),
                azure_deployment=llm_config.get("model", "gpt-5"),
            )
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=llm_config.get("api_key", ""),
                base_url=llm_config.get("base_url", ""),
                model=llm_config.get("model", "qwen-max"),
            )

        raw_responses = backend_mod.generate_cqs(llm, prompt_template, input_batches)
        cqs = backend_mod.parse_cqs(raw_responses)
        log(f"[LLM4KE] Generated {len(cqs)} CQs")

        # Compute entity coverage
        coverage_stats = {}
        try:
            from evaluation.coverage import load_entities_for_dataset, compute_string_coverage
            entities = load_entities_for_dataset(dataset_name)
            if entities:
                coverage_stats = compute_string_coverage(entities, cqs)
                log(f"[LLM4KE] Entity coverage: {coverage_stats['coverage_rate']:.2f}% "
                    f"({coverage_stats['covered_entities']}/{coverage_stats['total_entities']})")
        except Exception as e:
            log(f"[LLM4KE] Coverage computation failed: {e}")

        duration = time.time() - start
        return RunResult(
            method=self.name,
            dataset=dataset_name,
            generated_cqs=cqs,
            metrics={"coverage_stats": coverage_stats},
            intermediate_logs=logs,
            duration_seconds=duration,
        )
