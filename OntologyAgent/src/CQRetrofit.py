import asyncio
import json
import os
import random
import re
import sys
from copy import deepcopy
from datetime import datetime
from typing import Callable, Optional

_META_PREFIXES = ("owl:", "rdf:", "rdfs:", "xsd:")

from roles.ontology_segmenter import OntologySegmenter
from roles.competency_question_generator import CQGenerator
from roles.sparql_evaluator import SPARQLEvaluator
from config.Config import (
    DATA_DIR_PATH,
    ONTOLOGY_NAME,
    CQ_EXAMPLES_NUM,
    MAX_LOOP_COUNT,
    MAX_NUM_SUB_TRIPLES,
    SEGMENTATION_METHOD,
    RANDOM_SEED,
    LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_API_TYPE,
)


def _save_json(path: str, data) -> None:
    """Write *data* as pretty-printed JSON to *path*, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


async def main(
    data_dir: str = DATA_DIR_PATH,
    ontology_name: str = ONTOLOGY_NAME,
    cq_examples_num: int = CQ_EXAMPLES_NUM,
    max_loop_count: int = MAX_LOOP_COUNT,
    max_sub_triples: int = MAX_NUM_SUB_TRIPLES,
    segmentation_method: str = SEGMENTATION_METHOD,
    random_seed: int = RANDOM_SEED,
    results_dir: str = "results",
    progress_callback: Optional[Callable[[str], None]] = None,
    llm_config: Optional[dict] = None,
    skip_segmentation: bool = False,
    skip_validator: bool = False,
):
    # --- reproducibility -------------------------------------------------
    random.seed(random_seed)

    # --- logging helper --------------------------------------------------
    def log(msg: str):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    # --- resolve LLM model name for output paths -------------------------
    llm_model_name = LLM_MODEL
    if llm_config and "model" in llm_config:
        llm_model_name = llm_config["model"]

    # --- run output directory --------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        results_dir, ontology_name, llm_model_name, f"run_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # Save config snapshot
    run_config = {
        "data_dir": data_dir,
        "ontology_name": ontology_name,
        "cq_examples_num": cq_examples_num,
        "max_loop_count": max_loop_count,
        "max_sub_triples": max_sub_triples,
        "segmentation_method": segmentation_method,
        "random_seed": random_seed,
        "llm_model": llm_model_name,
        "results_dir": results_dir,
        "timestamp": timestamp,
        "skip_segmentation": skip_segmentation,
        "skip_validator": skip_validator,
    }
    _save_json(os.path.join(run_dir, "config.json"), run_config)

    # --- Build MetaGPT Config if llm_config provided ---------------------
    metagpt_config = None
    if llm_config:
        from metagpt.config2 import Config
        metagpt_config = Config.from_llm_config(llm_config)

    # 0. 读取输入
    json_file_path = os.path.join(data_dir, ontology_name, ontology_name + ".json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        ontology_info = json.load(f)

    segmenter = OntologySegmenter(
        max_graph_triples_num=max_sub_triples,
        segmentation_method=segmentation_method,
        random_seed=random_seed,
    )
    generator = CQGenerator()
    evaluator = SPARQLEvaluator()

    # Apply LLM config to roles if provided
    if metagpt_config:
        segmenter.config = metagpt_config
        generator.config = metagpt_config
        evaluator.config = metagpt_config

    ontology_file_path = os.path.join(
        data_dir, ontology_name, ontology_info["file_name"]
    )
    ontology_description = ontology_info["description"]
    ground_truth_cqs = ontology_info["competency_questions"]
    ground_truth_cq_examples = random.sample(
        ground_truth_cqs, min(len(ground_truth_cqs), cq_examples_num)
    )

    # 1. 解析 OWL 文件，形成本体三元组片段
    if skip_segmentation:
        log("Step 1: Parsing ontology file (segmentation SKIPPED - ablation mode)...")
        # Parse OWL but skip segmentation: use all triples as one chunk
        from roles.ontology_segmenter import LoadOWL
        loader = LoadOWL()
        parsed_ontology_dict = await loader.run(ontology_file_path)
        ontology_info["namespaces"] = parsed_ontology_dict["namespaces"]
        ontology_info["triples"] = parsed_ontology_dict["triples"]
        # Create a single chunk containing all triples
        chunks_list = [{
            "subgraph_id": 1,
            "namespaces": parsed_ontology_dict["namespaces"],
            "triples": parsed_ontology_dict["triples"],
        }]
        selected_tool = None
        log(
            f"Parsed {len(ontology_info['triples'])} triples, "
            f"using as single chunk (no segmentation)."
        )
        # Save segmentation metadata
        segmentation_meta = {
            "method": "none (skip_segmentation=True)",
            "random_seed": random_seed,
            "max_sub_triples": max_sub_triples,
            "total_triples": len(ontology_info["triples"]),
            "num_subgraphs": 1,
            "subgraph_sizes": [len(ontology_info["triples"])],
        }
        _save_json(os.path.join(run_dir, "segmentation.json"), segmentation_meta)
    else:
        log("Step 1: Parsing and segmenting ontology file...")
        task = "parse owl file into triples chunks"
        segmenter.set_file_path(ontology_file_path)
        await segmenter.run(task)
        parsed_ontology_dict = segmenter.get_ontology_dict()
        chunks_list = segmenter.get_segmented_chunks_list()
        ontology_info["namespaces"] = parsed_ontology_dict["namespaces"]
        ontology_info["triples"] = parsed_ontology_dict["triples"]

        # Report LLM tool selection if in "auto" mode
        selected_tool = segmenter.get_selected_tool()
        if selected_tool:
            log(
                f"LLM selected segmentation algorithm: {selected_tool['method']} "
                f"with args {selected_tool['args']}"
            )

        log(
            f"Parsed {len(ontology_info['triples'])} triples, "
            f"segmented into {len(chunks_list)} subgraphs."
        )

        # Save segmentation metadata
        segmentation_meta = {
            "method": segmentation_method,
            "random_seed": random_seed,
            "max_sub_triples": max_sub_triples,
            "total_triples": len(ontology_info["triples"]),
            "num_subgraphs": len(chunks_list),
            "subgraph_sizes": [len(c["triples"]) for c in chunks_list],
        }
        if selected_tool:
            segmentation_meta["llm_selected_tool"] = selected_tool
        _save_json(os.path.join(run_dir, "segmentation.json"), segmentation_meta)

    # 2-5. 处理每个子图，生成能力问题并评估覆盖率
    effective_max_loop = 1 if skip_validator else max_loop_count

    for i in range(len(chunks_list)):
        chunk_info = chunks_list[i]
        optimal_chunk_info = deepcopy(chunk_info)
        min_uncovered_entities_count = sys.maxsize

        # 计算子图中的所有实体（排除 meta 前缀）
        all_entities = set()
        for triple in chunk_info["triples"]:
            all_entities.add(triple[0])  # subject
            all_entities.add(triple[2])  # object
        all_entities = {e for e in all_entities if not any(e.startswith(p) for p in _META_PREFIXES)}
        total_entities = len(all_entities)

        log(f"\n{'='*80}")
        log(f"Processing subgraph {i+1}/{len(chunks_list)}")
        log(
            f"Subgraph contains {len(chunk_info['triples'])} triples, "
            f"{total_entities} entities"
        )
        if skip_validator:
            log(f"(Validator SKIPPED - ablation mode, single iteration only)")
        log(f"{'='*80}")

        for iteration in range(effective_max_loop):
            # 2. 生成能力问题
            log(
                f"\n[Subgraph {i+1}] Iteration {iteration+1}/{effective_max_loop}: "
                f"Generating CQs..."
            )
            task = "generate competency questions"
            generator.set_input(
                chunk_info, ontology_description, ground_truth_cq_examples
            )
            await generator.run(task)
            chunk_info = generator.get_chunk_info()
            optimal_chunk_info = deepcopy(chunk_info)

            if skip_validator:
                # Skip SPARQL evaluation entirely; estimate coverage via string matching
                log(
                    f"[Subgraph {i+1}] Iteration {iteration+1}/{effective_max_loop}: "
                    f"Validator skipped, using string-based coverage estimate..."
                )
                # String-based coverage: check if normalized entity names appear in CQ text
                cq_list = chunk_info.get("competency_questions", [])
                cq_texts_lower = [cq.lower() for cq in cq_list]
                covered_ents = []
                uncovered_ents = []
                for ent in all_entities:
                    # Normalize: strip prefix, split camelCase, lowercase
                    norm = ent.split(":", 1)[-1] if ":" in ent else ent
                    norm = norm.rsplit("#", 1)[-1]
                    norm = norm.rsplit("/", 1)[-1]
                    norm = re.sub(r"([a-z])([A-Z])", r"\1 \2", norm)
                    norm = re.sub(r"[_\-]", " ", norm).lower().strip()
                    if not norm or any(norm in cq_text for cq_text in cq_texts_lower):
                        covered_ents.append(ent)
                    else:
                        uncovered_ents.append(ent)
                chunk_info["uncovered_entities"] = uncovered_ents
                # In no-validator mode, treat all generated CQs as qualified
                chunk_info["qualified_questions"] = list(cq_list)
            else:
                # 3. 评估能力问题的覆盖率，如果没有覆盖到所有实体，则继续扩展能力问题
                log(
                    f"[Subgraph {i+1}] Iteration {iteration+1}/{effective_max_loop}: "
                    f"Evaluating SPARQL coverage..."
                )
                task = "evaluate the coverage of competency questions"
                evaluator.set_chunk_info(chunk_info)
                await evaluator.run(task)
                chunk_info = evaluator.get_chunk_info()

            # 4. 计算覆盖率并输出
            uncovered_entities_count = len(chunk_info["uncovered_entities"])
            covered_entities_count = total_entities - uncovered_entities_count
            coverage_rate = (
                (covered_entities_count / total_entities * 100)
                if total_entities > 0
                else 0
            )

            log(f"\nIteration {iteration+1} results:")
            log(f"  - Generated CQs: {len(chunk_info['competency_questions'])}")
            log(
                f"  - Qualified CQs: "
                f"{len(chunk_info.get('qualified_questions', []))}"
            )
            log(f"  - Covered entities: {covered_entities_count}/{total_entities}")
            log(f"  - Uncovered entities: {uncovered_entities_count}")
            log(f"  - Coverage rate: {coverage_rate:.2f}%")

            # Save per-subgraph per-iteration intermediate results
            iter_snapshot = {
                "subgraph_id": i,
                "iteration": iteration,
                "competency_questions": chunk_info.get("competency_questions", []),
                "qualified_questions": chunk_info.get("qualified_questions", []),
                "uncovered_entities": chunk_info.get("uncovered_entities", []),
                "coverage": {
                    "total_entities": total_entities,
                    "covered_entities": covered_entities_count,
                    "uncovered_entities_count": uncovered_entities_count,
                    "coverage_rate": coverage_rate,
                },
            }
            _save_json(
                os.path.join(
                    run_dir, f"subgraph_{i}_iter_{iteration}.json"
                ),
                iter_snapshot,
            )

            # 5. 更新最优的能力问题结果
            if uncovered_entities_count < min_uncovered_entities_count:
                min_uncovered_entities_count = uncovered_entities_count
                optimal_chunk_info = deepcopy(chunk_info)
            if uncovered_entities_count == 0:
                log(f"  100% coverage reached, stopping early.")
                break

        # 记录最终覆盖率信息到chunk_info中
        optimal_chunk_info["coverage_stats"] = {
            "total_entities": total_entities,
            "covered_entities": total_entities
            - len(optimal_chunk_info["uncovered_entities"]),
            "uncovered_entities_count": len(
                optimal_chunk_info["uncovered_entities"]
            ),
            "coverage_rate": (
                (
                    (
                        total_entities
                        - len(optimal_chunk_info["uncovered_entities"])
                    )
                    / total_entities
                    * 100
                )
                if total_entities > 0
                else 0
            ),
        }
        chunks_list[i] = optimal_chunk_info

        log(f"\nSubgraph {i+1} final results:")
        log(
            f"  - Best coverage: "
            f"{optimal_chunk_info['coverage_stats']['coverage_rate']:.2f}%"
        )
        log(
            f"  - Covered entities: "
            f"{optimal_chunk_info['coverage_stats']['covered_entities']}"
            f"/{total_entities}"
        )
        log(f"{'='*80}\n")

    # 6. 输出结果并计算总体覆盖率统计
    ontology_info["retrofitted_competency_questions"] = list()
    for item in chunks_list:
        ontology_info["retrofitted_competency_questions"].extend(
            item["competency_questions"]
        )
    ontology_info["subgraph_chunks"] = chunks_list

    # Build per-iteration average coverage history from snapshot files
    iteration_coverage_history = []
    # Track last known coverage per subgraph (for early-stop carry-forward)
    last_coverage_per_subgraph = {}
    for iteration in range(max_loop_count):
        iter_coverages = []
        for sg_idx in range(len(chunks_list)):
            snapshot_path = os.path.join(
                run_dir, f"subgraph_{sg_idx}_iter_{iteration}.json"
            )
            if os.path.exists(snapshot_path):
                with open(snapshot_path, "r", encoding="utf-8") as sf:
                    snap = json.load(sf)
                last_coverage_per_subgraph[sg_idx] = snap["coverage"]
            # Use last known coverage (handles early-stop subgraphs)
            if sg_idx in last_coverage_per_subgraph:
                iter_coverages.append(last_coverage_per_subgraph[sg_idx])
        if iter_coverages:
            avg_rate = sum(
                c["coverage_rate"] for c in iter_coverages
            ) / len(iter_coverages)
            iteration_coverage_history.append({
                "iteration": iteration,
                "per_subgraph": iter_coverages,
                "avg_coverage_rate": round(avg_rate, 2),
            })
    ontology_info["iteration_coverage_history"] = iteration_coverage_history

    # 计算总体覆盖率统计（全局去重，排除 meta 前缀）
    global_entities = set()
    for chunk in chunks_list:
        for triple in chunk.get("triples", []):
            global_entities.add(triple[0])
            global_entities.add(triple[2])
    global_entities = {e for e in global_entities if not any(e.startswith(p) for p in _META_PREFIXES)}

    global_covered = set()
    for chunk in chunks_list:
        uncovered = set(chunk.get("uncovered_entities", []))
        chunk_ents = set()
        for triple in chunk.get("triples", []):
            chunk_ents.add(triple[0])
            chunk_ents.add(triple[2])
        chunk_ents = {e for e in chunk_ents if not any(e.startswith(p) for p in _META_PREFIXES)}
        global_covered |= (chunk_ents - uncovered)

    total_entities_all = len(global_entities)
    total_covered_all = len(global_covered & global_entities)
    overall_coverage_rate = (
        (total_covered_all / total_entities_all * 100)
        if total_entities_all > 0
        else 0
    )

    ontology_info["overall_coverage_stats"] = {
        "total_subgraphs": len(chunks_list),
        "total_entities": total_entities_all,
        "total_covered_entities": total_covered_all,
        "total_uncovered_entities": total_entities_all - total_covered_all,
        "overall_coverage_rate": overall_coverage_rate,
        "subgraph_coverage_details": [
            {
                "subgraph_id": idx,
                "total_entities": chunk["coverage_stats"]["total_entities"],
                "covered_entities": chunk["coverage_stats"]["covered_entities"],
                "coverage_rate": chunk["coverage_stats"]["coverage_rate"],
            }
            for idx, chunk in enumerate(chunks_list)
        ],
    }

    log(f"\n{'='*80}")
    log(f"Overall coverage statistics:")
    log(f"  - Subgraphs: {len(chunks_list)}")
    log(f"  - Total entities: {total_entities_all}")
    log(f"  - Covered entities: {total_covered_all}")
    log(f"  - Uncovered entities: {total_entities_all - total_covered_all}")
    log(f"  - Overall coverage: {overall_coverage_rate:.2f}%")
    log(
        f"  - Total CQs generated: "
        f"{len(ontology_info['retrofitted_competency_questions'])}"
    )
    log(f"{'='*80}\n")

    # Save final aggregated results to run directory
    _save_json(os.path.join(run_dir, "results.json"), ontology_info)
    log(f"Results saved to {os.path.join(run_dir, 'results.json')}")

    # Backward-compatible copy for analysis scripts
    compat_path = os.path.join(
        data_dir, ontology_name, f"results_{llm_model_name}.json"
    )
    _save_json(compat_path, ontology_info)
    log(f"Backward-compatible results saved to {compat_path}")

    return ontology_info


if __name__ == "__main__":
    asyncio.run(main())
