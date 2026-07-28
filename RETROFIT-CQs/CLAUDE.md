# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RETROFIT-CQs** is a research project for retrofitting Competency Questions (CQs) from existing ontologies using LLMs. CQs define the scope of an ontology and are often missing from published ontologies. The pipeline: (1) extract triples from ontologies, (2) feed triples to LLMs to generate CQs, (3) evaluate generated CQs against benchmarks using SBERT similarity matching.

Published at SAC '24 (ACM SIGAPP). Contact: Reham Alharbi (r.alharbi@liverpool.ac.uk).

## Architecture

The project is a single-command pipeline. All modules live at the project root:

```
run.py              # Entry point: python run.py --ontology X --llm Y
config.py           # CLI args, .env loading, format detection, run dir setup
extract_triples.py  # Stage 1: Triple extraction (rdflib default, DeepOnto optional)
generate_cqs.py     # Stage 2: Unified LLM interface + CQ generation
evaluate_cqs.py     # Stage 3: SBERT matching + precision/recall/F1
label_cqs.py        # Optional: CQ simple/complex classification
```

### Stage 1: Triple Extraction (`extract_triples.py`)
- `extract_triples_rdflib()` — Default. Uses rdflib to parse OWL/TTL/RDF, extracts local names, filters with `_is_valid()` (letters/underscores, length > 2). Outputs tab-delimited CSV.
- `extract_triples_deeponto()` — Optional (`--use-deeponto`). Uses DeepOnto's `OntologyProjector`. Lazy-imports DeepOnto.
- Input: OWL/TTL/RDF files from `Data/Ontologies/`
- Output: `outputs/<timestamp>/triples.csv`

### Stage 2: CQ Generation (`generate_cqs.py`)
Three LLM backends behind a unified `LLMClient` ABC with a single `complete(system_prompt, user_prompt)` method:
- `AzureOpenAIClient` (`--llm gpt4`) — Azure OpenAI via `openai.ChatCompletion.create(engine=...)`
- `OllamaClient` (`--llm llama`) — Ollama via `langchain_ollama.ChatOllama`
- `QwenClient` (`--llm qwen`) — DashScope via `openai.ChatCompletion.create(model=...)`
- Factory function `create_llm_client(name)` instantiates the right backend.
- `generate_questions_from_csv()` reads triples, calls LLM per row, writes Excel.
- Output: `outputs/<timestamp>/generated_cqs.xlsx`

### Stage 3: Evaluation (`evaluate_cqs.py`)
- `load_benchmark_cqs()` — Reads `Data/CQs/Only CQs.xlsx`, auto-detects ontology URI from the ontology file to filter benchmark CQs by the `Ontology` column. Falls back to all CQs if no match. Supports `--benchmark-filter` override.
- `extract_generated_cqs()` — Splits multi-line CQ answers, strips numbering prefixes.
- `run_sbert_matching()` — Reproduces `sbert_final_2.py` logic: for each benchmark CQ, finds generated CQs with cosine similarity >= threshold, removes matched CQs from pool.
- `compute_metrics()` — Computes precision/recall/F1, saves `metrics.json`.
- Output: `outputs/<timestamp>/matched_cqs.xlsx`, `outputs/<timestamp>/metrics.json`

### Optional: CQ Labeling (`label_cqs.py`)
- Reuses `LLMClient` from `generate_cqs.py` (not limited to Qwen).
- Classifies CQs as "Simple" or "Complex" based on single-hop vs multi-hop criteria.
- Activated with `--label` flag.

### Data & Legacy
- `Data/Ontologies/` — Source ontology files
- `Data/CQs/` — Benchmark competency questions
- `Results/` — Historical results organized by ontology
- `Implementation/`, `Evaluation/`, `Extracting Triples/` — Original scripts preserved for reference

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in API keys
cp .env.example .env

# Full pipeline
python run.py --ontology Data/Ontologies/vicinitycore.owl --llm gpt4

# With custom threshold
python run.py --ontology Data/Ontologies/videogameontology.owl --llm qwen --threshold 0.7

# Skip extraction, use existing triples
python run.py --skip-extract --triples-csv Data/ExtractingTriples/base_ontology.csv --llm llama --ontology Data/Ontologies/base_ontology.owl

# Evaluation only
python run.py --skip-extract --skip-generate \
  --triples-csv Data/ExtractingTriples/base_ontology.csv \
  --generated-cqs Implementation/MyResults/base_ontology_gpt.xlsx \
  --ontology Data/Ontologies/base_ontology.owl --llm gpt4

# With optional CQ labeling
python run.py --ontology Data/Ontologies/vicinitycore.owl --llm qwen --label
```

All outputs go to `outputs/<timestamp>/`: `triples.csv`, `generated_cqs.xlsx`, `matched_cqs.xlsx`, `metrics.json`, `run.log`, `run_config.json`.

## Key Dependencies

- `python-dotenv` — Environment variable loading from `.env`
- `rdflib` — Ontology parsing (Stage 1, default)
- `deeponto` — Ontology parsing (Stage 1, optional with `--use-deeponto`)
- `openai` (<1.0) — Azure OpenAI and DashScope API calls (Stage 2)
- `langchain-ollama` — Ollama integration (Stage 2)
- `sentence-transformers` — SBERT model for CQ matching (Stage 3)
- `pandas`, `openpyxl` — Data I/O
- `torch` — Required by sentence-transformers

## Important Conventions

- All CSV files use **tab delimiters** (`\t`)
- API keys are loaded from `.env` (never hardcoded). See `.env.example` for required variables.
- The CQ generation system prompt is consistent across all LLM backends (defined once in `generate_cqs.py:SYSTEM_PROMPT`)
- SBERT model: `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- Benchmark CQ counts are **auto-detected** from the ontology URI (no hardcoded constants)
- Triple validity filter: `re.match(r'^[a-zA-Z_]+$', value) and len(value) > 2`
