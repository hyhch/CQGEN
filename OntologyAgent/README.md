# CQGen-MAS (OntologyAgent)

This directory contains the **CQGen-MAS** core: the three agents (`roles/`), the pipeline driver (`src/CQRetrofit.py`), the batch runner (`run_experiments.py`), evaluation utilities (`utils/`), the evaluation ontologies (`dataset/`), and the human-evaluation annotations (`human_eval/`).

> For the full repository overview, installation, and the multi-LLM experiment grid, see the **[top-level README](../README.md)**.

## Quick start

```bash
pip install -r requirements.txt          # METIS C library also required for the 'metis' option
# edit config/Config.py: set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
python run_experiments.py --all          # all datasets with the configured LLM
```

Key flags: `--dataset <name>`, `--all-datasets`, `--model <name>`, `--segmentation {auto,metis,louvain,leiden,spectral}`, `--ablation segmentation`, `--seed <int>`.
