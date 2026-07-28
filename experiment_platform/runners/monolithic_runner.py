"""Runner for Monolithic CQ generation baseline (ablation experiment).

Combines ontology analysis, CQ generation, and SPARQL validation into a single
LLM call — no segmentation, no iterative feedback loop, no multi-agent
collaboration. This serves as a baseline to demonstrate the value of the MAS
architecture in CQGen-MAS.
"""

import json
import os
import random
import re
import time
from typing import Callable, Optional

from .base import BaseRunner, RunResult

# Workspace root
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OA_ROOT = os.path.join(_WORKSPACE, "OntologyAgent")


SYSTEM_PROMPT = """\
You are an expert in ontology engineering. Your task is to analyze a given ontology, \
generate competency questions (CQs), and validate them — all in a single response.

A competency question outlines the scope of an ontology and provides an idea about \
the knowledge that needs to be entailed in the ontology."""

USER_PROMPT_TEMPLATE = """\
# Task
Analyze the following ontology triples, generate as many competency questions as \
possible, and self-validate each question against the ontology structure.

# Ontology Description
{description}

# Ontology Triples
{triples}

# Example Competency Questions
{examples}

# Question Complexity Guidelines
1. [Simple]: Single-class/property retrieval and direct triple patterns requiring \
no reasoning.
   - "What is the definition of Mammal?"
   - "Who founded Apple Inc.?"

2. [Complex]: Multi-hop queries (topological distance >= 2) that traverse multiple \
ontology classes, properties, or relations.
   - "Which companies were founded by Stanford alumni?"
   - "Which cities in California have populations over 1 million?"

# Instructions
1. First, analyze the ontology structure — identify key classes, properties, and \
relationships.
2. Generate as many competency questions as possible covering both Simple and \
Complex types.
3. For each generated question, mentally verify whether a corresponding SPARQL \
query could be constructed from the given triples. Discard questions that cannot \
be grounded in the ontology.
4. Output ONLY the final validated questions in the format below. Do NOT include \
any explanations, analysis, or commentary.

# Output Format
```questions
1. [Simple] Question 1?
2. [Complex] Question 2?
3. [Simple] Question 3?
```"""


def _decode_cqs(response: str) -> list:
    """Parse CQs from the LLM response."""
    pattern = re.compile(r"(\d+)\.\s*\[(Simple|Complex)\]\s*(.*?)\s*(?:\n|$)")
    cqs = []
    for match in pattern.finditer(response):
        label = match.group(2)
        question = match.group(3).strip()
        if question:
            cqs.append(f"[{label}] {question}")
    # Fallback: try numbered lines without labels
    if not cqs:
        for line in response.splitlines():
            line = line.strip()
            m = re.match(r"^\d+\.\s*(.+)", line)
            if m:
                q = m.group(1).strip()
                if q and len(q) > 10:
                    cqs.append(q)
    return cqs


def _load_ontology_data(dataset_name):
    """Load ontology triples, description, and example CQs for a dataset."""
    from dataset_registry import get_dataset_info
    info = get_dataset_info(dataset_name)
    oa_info = info["ontology_agent"]
    data_dir = oa_info["data_dir"]
    ontology_name = oa_info["ontology_name"]

    json_path = os.path.join(data_dir, ontology_name, f"{ontology_name}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        ontology_info = json.load(f)

    # Parse the OWL file to get triples
    ontology_file_path = os.path.join(
        data_dir, ontology_name, ontology_info["file_name"]
    )

    from rdflib import Graph, URIRef
    g = Graph()
    ext = os.path.splitext(ontology_file_path)[1].lower()
    fmt_map = {".owl": "xml", ".rdf": "xml", ".ttl": "turtle"}
    g.parse(ontology_file_path, format=fmt_map.get(ext, "xml"))

    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            triples.append((
                str(s.n3(namespace_manager=g.namespace_manager)),
                str(p.n3(namespace_manager=g.namespace_manager)),
                str(o.n3(namespace_manager=g.namespace_manager)),
            ))

    return {
        "triples": triples,
        "description": ontology_info.get("description", ""),
        "ground_truth_cqs": ontology_info.get("competency_questions", []),
    }


def _build_llm_client(llm_config):
    """Build an OpenAI-compatible LLM client."""
    api_type = llm_config.get("api_type", "openai")

    if api_type == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=llm_config.get("api_key", "no-key"),
            azure_endpoint=llm_config.get("base_url", ""),
            api_version=llm_config.get("api_version", "2024-12-01-preview"),
        )
    else:
        from openai import OpenAI
        return OpenAI(
            api_key=llm_config.get("api_key", "no-key"),
            base_url=llm_config.get("base_url", ""),
        )


def _chunk_triples(triples, max_chunk_size):
    """Split triples into chunks of at most max_chunk_size."""
    chunks = []
    for i in range(0, len(triples), max_chunk_size):
        chunks.append(triples[i:i + max_chunk_size])
    return chunks


class MonolithicRunner(BaseRunner):
    name = "Monolithic"

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

        cq_examples_num = params.get("cq_examples_num", 10)
        use_chunking = params.get("use_chunking", False)
        chunk_size = params.get("chunk_size", 25)
        random_seed = params.get("random_seed", 42)

        log(f"[Monolithic] Running on {dataset_name}")
        log(f"[Monolithic] Mode: {'Chunked' if use_chunking else 'Full'}")

        # Load ontology data
        data = _load_ontology_data(dataset_name)
        triples = data["triples"]
        description = data["description"]
        gt_cqs = data["ground_truth_cqs"]

        log(f"[Monolithic] Loaded {len(triples)} triples")

        # Sample few-shot examples
        rng = random.Random(random_seed)
        examples = rng.sample(gt_cqs, min(len(gt_cqs), cq_examples_num))
        examples_text = "\n".join(f"- {q}" for q in examples)

        # Build LLM client
        client = _build_llm_client(llm_config)
        model = llm_config.get("model", "qwen-max")

        all_cqs = []

        if use_chunking:
            # Chunked mode: split triples and call LLM per chunk
            chunks = _chunk_triples(triples, chunk_size)
            log(f"[Monolithic] Split into {len(chunks)} chunks (max {chunk_size} triples each)")

            for idx, chunk in enumerate(chunks):
                log(f"[Monolithic] Processing chunk {idx+1}/{len(chunks)} ({len(chunk)} triples)...")
                triples_text = "\n".join(f"({s}, {p}, {o})" for s, p, o in chunk)
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    description=description,
                    triples=triples_text,
                    examples=examples_text,
                )
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    content = response.choices[0].message.content.strip()
                    cqs = _decode_cqs(content)
                    all_cqs.extend(cqs)
                    log(f"[Monolithic] Chunk {idx+1}: generated {len(cqs)} CQs")
                except Exception as e:
                    log(f"[Monolithic] Chunk {idx+1} failed: {e}")
        else:
            # Full mode: single LLM call with all triples
            triples_text = "\n".join(f"({s}, {p}, {o})" for s, p, o in triples)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                description=description,
                triples=triples_text,
                examples=examples_text,
            )

            log(f"[Monolithic] Sending single prompt with {len(triples)} triples...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content.strip()
                all_cqs = _decode_cqs(content)
                log(f"[Monolithic] Generated {len(all_cqs)} CQs")
            except Exception as e:
                log(f"[Monolithic] Generation failed: {e}")

        log(f"[Monolithic] Total CQs: {len(all_cqs)}")

        # Compute entity coverage
        coverage_stats = {}
        try:
            from evaluation.coverage import load_entities_for_dataset, compute_string_coverage
            entities = load_entities_for_dataset(dataset_name)
            if entities:
                coverage_stats = compute_string_coverage(entities, all_cqs)
                log(f"[Monolithic] Entity coverage: {coverage_stats['coverage_rate']:.2f}% "
                    f"({coverage_stats['covered_entities']}/{coverage_stats['total_entities']})")
        except Exception as e:
            log(f"[Monolithic] Coverage computation failed: {e}")

        duration = time.time() - start
        return RunResult(
            method=self.name,
            dataset=dataset_name,
            generated_cqs=all_cqs,
            metrics={"coverage_stats": coverage_stats},
            intermediate_logs=logs,
            duration_seconds=duration,
        )
