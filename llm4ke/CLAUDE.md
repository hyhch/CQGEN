# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM4KE (Large Language Models for Knowledge Engineering) is a research tool that uses LLMs to generate **Competency Questions (CQs)** from OWL ontologies, then evaluates the generated CQs against human-authored ground truth using semantic similarity. It accompanies ESWC 2024 and ISWC 2024 papers by EURECOM researchers.

## Commands

All commands use the unified CLI entry point `llm4ke.py`. Run from repo root with the `ontology` conda environment.

### Generate Competency Questions
```shell
conda run -n ontology python llm4ke.py generate <OntologyName> --task <task> --llm <backend>

# Example:
conda run -n ontology python llm4ke.py generate Odeuropa --task all_classes --llm qwen
```
Output is auto-saved to `data_out/<OntologyName>/<mode>/<OntologyName>_<llm>_<n_examples>.txt`.

### Evaluate Generated CQs
```shell
conda run -n ontology python llm4ke.py evaluate <OntologyName|all> [--threshold 0.6] [--labels src/CQs_labeled.xlsx]

# Example:
conda run -n ontology python llm4ke.py evaluate Odeuropa --threshold 0.8
```
Results are saved to `results_<name>.json`.

### Generate + Evaluate in One Step
```shell
conda run -n ontology python llm4ke.py run <OntologyName> --task <task> --llm <backend>
```

### Install Dependencies
```shell
conda activate ontology
pip install -r requirements.txt
```

### Convert CQs from Excel to YAML
```shell
python transfer_CQs.py  # reads CQs.xlsx, writes to data/<ontology>/cqs/cqs.yml
```

## Architecture

### Unified CLI (`llm4ke.py`)

Single entry point with three subcommands: `generate`, `evaluate`, `run` (both).

### Source Modules (`src/`)

- `ontology.py` — Ontology loading (rdflib), class/property/schema extraction, ground truth loading, batching
- `llm_backend.py` — Unified LLM API backend via LangChain ChatOpenAI (supports any OpenAI-compatible API)
- `evaluator.py` — Semantic similarity evaluation (P/R/F1) with optional simple/complex classification stats
- `LocalTemplate.py` — YAML-based prompt template loader with import/substitution support
- `prompt_templates/*.yml` — Three prompt modes: `all_classes`, `all_classes+properties`, `logic`

### Configuration (`config.yml`)

Defines LLM backends. Each backend specifies `base_url`, `model`, and `api_key_env` (environment variable name for the API key). Currently configured: `qwen` (DashScope), `gpt4o` (Azure), `llama3` (Ollama remote).

### Data Layout

Each ontology in `data/<OntologyName>/`:
- `dm/` — OWL/RDF/TTL ontology files
- `cqs/cqs.yml` — Ground truth competency questions
- `description.txt` — (optional) ontology description for prompt enrichment

### Output Directory (`data_out/`)

Generated CQ files: `data_out/<OntologyName>/<mode>/<OntologyName>_<llm>_<examples>.txt`
Raw LLM responses: `data_out/<OntologyName>/<mode>/<OntologyName>_<llm>_<examples>_raw.json`

## Important Notes

- Must run from repo root (prompt template paths are relative)
- API keys are read from environment variables (see `config.yml` for variable names)
- The evaluation expects filenames in `<OntologyName>_<llm>_<examples>.txt` format
- Generation automatically saves to the correct evaluation directory structure
