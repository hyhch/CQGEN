# CQGen-MAS

**Iterative Multi-Agent CQ Generation for Ontology Retrofitting**
*(ISWC 2026 — camera-ready companion code)*

CQGen-MAS is an LLM-powered multi-agent system that retrofits existing ontologies with **competency questions (CQs)**. It runs an iterative generation–validation loop in which a **Segmenter** adaptively partitions the ontology into coherent subgraphs, a **Generator** formulates candidate CQs, and a **SPARQL Validator** verifies them against the ontology and feeds uncovered entities back to the Generator until sufficient entity coverage is reached.

---

## Repository structure

| Directory | Description |
|-----------|-------------|
| [`OntologyAgent/`](OntologyAgent/) | **CQGen-MAS core.** The three agents (`roles/`), the pipeline driver (`src/CQRetrofit.py`), the batch runner (`run_experiments.py`), evaluation utilities (`utils/`), the four evaluation ontologies (`dataset/`), and the human-evaluation annotations (`human_eval/`). |
| [`experiment_platform/`](experiment_platform/) | **Orchestration layer.** Multi-LLM × multi-dataset experiment grid, ablation and coverage runners, shared dataset registry, LLM configuration (`llm_configs.json`), and the Validator-on-ground-truth analysis. |
| [`llm4ke/`](llm4ke/) | **LLM4KE** baseline — Rebboud et al., *“Can LLMs Generate Competency Questions?”*, ESWC 2024. |
| [`RETROFIT-CQs/`](RETROFIT-CQs/) | **RETROFIT-CQs** baseline — Alharbi et al., *“An Experiment in Retrofitting Competency Questions for Existing Ontologies”*, SAC 2024. |

---

## Requirements

- Python ≥ 3.11
- An OpenAI-compatible LLM endpoint (tested with Qwen3-Max, GLM-5, and GPT-5)
- The **METIS** C library (for the METIS partitioning option): install e.g. with `apt install libmetis-dev` or build from <https://github.com/KarypisLab/METIS>.

Install Python dependencies:

```bash
pip install -r OntologyAgent/requirements.txt
```

> The pipeline is built on the [MetaGPT](https://github.com/geekan/MetaGPT) framework, which is pulled in as a dependency. Baseline directories (`llm4ke/`, `RETROFIT-CQs/`) ship their own `requirements.txt` for their additional dependencies.

---

## Configuration

All LLM credentials are **placeholders** (`YOUR_API_KEY_HERE`) — replace them before running.

- **CQGen-MAS (single-model runs):** edit `OntologyAgent/config/Config.py`.
  - `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` — your endpoint.
  - `ONTOLOGY_NAME`, `SEGMENTATION_METHOD` (`auto` | `metis` | `louvain` | `leiden` | `spectral`), `MAX_LOOP_COUNT`, `CQ_EXAMPLES_NUM`, `RANDOM_SEED`.
- **Full experiment grid (3 LLMs):** edit `experiment_platform/llm_configs.json`, which carries the per-model settings for `qwen-max`, `glm-5`, and `gpt-5`.

---

## Usage

### 1. Run CQGen-MAS

```bash
cd OntologyAgent
# single dataset with the configured LLM
python run_experiments.py --dataset videogameontology
# all datasets with the configured LLM
python run_experiments.py --all-datasets
# full grid: all datasets × all configured LLMs
python run_experiments.py --all
```

Useful flags: `--model <name>`, `--segmentation <algo>`, `--ablation segmentation` (compares segmentation algorithms), `--seed <int>`, `--results-dir <path>`. Datasets: `onem2m`, `saref4env`, `videogameontology`, `vicinitycore` (`demcare` is also available but is not used in the paper).

### 2. Run the baselines and the multi-LLM grid

The baselines and the full ablation/coverage analyses are orchestrated from `experiment_platform/`:

```bash
cd experiment_platform
# ablation study (configurations A/B/C or all)
python ablation_runner.py --config all
# main multi-LLM × multi-dataset experiment
python rts_experiment.py [--dataset <name>] [--model <name>] [--force]
# recompute coverage metrics from stored results
python recalc_coverage.py [--dry-run]
```

Per-baseline usage (prompts, inputs, outputs) is documented in each baseline's own README.

### 3. Validator-on-ground-truth analysis

To reproduce the Validator's pass rate on the 240 ground-truth CQs (Section 5, ablation):

```bash
cd experiment_platform
python validator_on_gt.py --model gpt-5 --all
```

### 4. Human evaluation

The human-evaluation data and annotation guidelines ship under `OntologyAgent/human_eval/` (`annotation_guidelines.txt`, `annotation_sheet.csv`, `annotated_sheet1.csv`, `annotated_sheet2.csv`, `answer_key.csv`). Aggregation utilities live in `OntologyAgent/utils/human_eval_*.py`.

---

## Datasets

The four evaluation ontologies are under `OntologyAgent/dataset/`:

| Directory | Ontology | Used in paper |
|-----------|----------|:---:|
| `onem2m/` | OneM2M | ✅ |
| `saref4env/` | SAREF4ENV | ✅ |
| `videogameontology/` | Video Game Ontology (VGO) | ✅ |
| `vicinitycore/` | Vicinity Core (VC) | ✅ |
| `demcare/` | DemCare | — (extra, not used in the paper) |

---

## Outputs

Experiment outputs are written to a `results/` directory (git-ignored). Each run stores its generated CQs, SPARQL translations, coverage statistics, and validation traces as JSON. Running the scripts regenerates all results reported in the paper.

---

## Reproducibility notes

- Each LLM independently executes the full CQGen-MAS pipeline; models are **not** mixed across agent roles. Tables in the paper report averages over the three LLMs.
- The five few-shot CQ examples are drawn from the CORAL corpus (Fernández-Izquierdo et al., 2019) using ontologies **other** than the target ontology, so there is no overlap with the evaluation ground truth.
- Set a fixed `RANDOM_SEED` (default `42`) for reproducible segmentation (for the `random` method) and few-shot sampling.

---

## Citation

If you use this code, please cite the ISWC 2026 paper:

```bibtex
@inproceedings{cqgenmas2026,
  title     = {CQGen-MAS: Iterative Multi-Agent CQ Generation for Ontology Retrofitting},
  author    = {Du, Fei and Chen, Li and Kharlamov, Evgeny and Lu, Yihan and Liu, Weidong},
  booktitle = {International Semantic Web Conference (ISWC)},
  year      = {2026}
}
```

The baselines retain their original licenses and citations (see `llm4ke/LICENSE` and `RETROFIT-CQs/`).
