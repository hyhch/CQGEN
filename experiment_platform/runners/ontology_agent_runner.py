"""Runner for OntologyAgent CQ generation method."""

import asyncio
import importlib.util
import os
import sys
import time
from typing import Callable, Optional

from .base import BaseRunner, RunResult

# Workspace root (parent of experiment_platform/)
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OA_ROOT = os.path.join(_WORKSPACE, "OntologyAgent")


def _load_cq_main():
    """Load OntologyAgent's CQRetrofit.main() via importlib to avoid 'src' namespace conflicts.

    OntologyAgent's ``src/`` directory has no ``__init__.py`` and collides
    with llm4ke's ``src/`` when both roots are on ``sys.path``.  By loading
    the module from its absolute file path we side-step the whole problem.

    We also need to:
    1. Set METAGPT_PROJECT_ROOT so MetaGPT finds OntologyAgent's config2.yaml.
    2. Import metagpt BEFORE adding OA_ROOT to sys.path, because OA_ROOT
       contains a local ``metagpt/`` directory (with only ``tools/``) that
       would shadow the real installed metagpt package.
    3. Add OA_ROOT to sys.path for CQRetrofit.py's own imports
       (``from roles.…``, ``from config.Config …``).
    """
    # Step 1: Configure MetaGPT to read OntologyAgent's config
    os.environ.setdefault("METAGPT_PROJECT_ROOT", _OA_ROOT)

    # Step 2: Import metagpt BEFORE OA_ROOT goes on sys.path
    import metagpt  # noqa: F401

    # Step 3: Now add OA_ROOT for roles/config imports
    if _OA_ROOT not in sys.path:
        sys.path.insert(0, _OA_ROOT)

    module_path = os.path.join(_OA_ROOT, "src", "CQRetrofit.py")
    spec = importlib.util.spec_from_file_location("oa_cq_retrofit", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


class OntologyAgentRunner(BaseRunner):
    name = "OntologyAgent"

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
        oa_info = info["ontology_agent"]
        data_dir = oa_info["data_dir"]
        ontology_name = oa_info["ontology_name"]

        segmentation_method = params.get("segmentation_method", "auto")
        max_iterations = params.get("max_iterations", 3)
        cq_examples_num = params.get("cq_examples_num", 10)
        max_sub_triples = params.get("max_sub_triples", 25)
        random_seed = params.get("random_seed", 42)
        skip_segmentation = params.get("skip_segmentation", False)
        skip_validator = params.get("skip_validator", False)

        log(f"[OntologyAgent] Running on {ontology_name}")
        log(f"[OntologyAgent] Segmentation: {segmentation_method}, Max iterations: {max_iterations}")
        if skip_segmentation:
            log(f"[OntologyAgent] ABLATION: Segmentation disabled")
        if skip_validator:
            log(f"[OntologyAgent] ABLATION: SPARQL Validator disabled")

        # Build llm_config for OntologyAgent (uses MetaGPT format)
        oa_llm_config = {
            "api_type": llm_config.get("api_type", "openai"),
            "model": llm_config.get("model", "qwen-max"),
            "api_key": llm_config.get("api_key", ""),
            "base_url": llm_config.get("base_url", ""),
        }
        if llm_config.get("api_version"):
            oa_llm_config["api_version"] = llm_config["api_version"]

        cq_main = _load_cq_main()

        # Run the async main function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                cq_main(
                    data_dir=data_dir,
                    ontology_name=ontology_name,
                    cq_examples_num=cq_examples_num,
                    max_loop_count=max_iterations,
                    max_sub_triples=max_sub_triples,
                    segmentation_method=segmentation_method,
                    random_seed=random_seed,
                    progress_callback=lambda msg: log(msg),
                    llm_config=oa_llm_config,
                    skip_segmentation=skip_segmentation,
                    skip_validator=skip_validator,
                )
            )
        finally:
            loop.close()

        # Extract generated CQs from result
        cqs = result.get("retrofitted_competency_questions", [])
        log(f"[OntologyAgent] Generated {len(cqs)} CQs")

        # Extract coverage stats
        coverage_stats = result.get("overall_coverage_stats", {})
        iteration_history = result.get("iteration_coverage_history", [])

        # Compute retroactive coverage if not present in result
        if not coverage_stats:
            from evaluation.coverage import compute_oa_coverage_from_results
            coverage_stats = compute_oa_coverage_from_results(result)

        if not iteration_history:
            from evaluation.coverage import compute_oa_iteration_coverage
            iteration_history = compute_oa_iteration_coverage(result)

        duration = time.time() - start
        return RunResult(
            method=self.name,
            dataset=dataset_name,
            generated_cqs=cqs,
            metrics={
                "coverage_stats": coverage_stats,
                "iteration_coverage_history": iteration_history,
            },
            intermediate_logs=logs,
            duration_seconds=duration,
        )
