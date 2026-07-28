"""
Human Evaluation Sampler for CQGen-MAS

Samples generated CQs into matched (TP) and unmatched (FP) groups based on
Sentence-BERT cosine similarity against ground truth, then exports a
shuffled annotation spreadsheet (CSV) for human evaluators.

Usage:
    python utils/human_eval_sampler.py [--n_unmatched 20] [--n_matched 10] [--threshold 0.6] [--seed 42]
"""

import json
import os
import csv
import random
import argparse
from collections import defaultdict

from rdflib import Graph as RDFGraph, URIRef, OWL, RDF
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(os.path.dirname(PROJECT_ROOT), "experiment_platform", "results")
DATASET_BASE = os.path.join(PROJECT_ROOT, "dataset")

DATASETS = {
    "onem2m":              {"gt_dir": "onem2m",              "onto_file": "onem2m.owl",              "label": "OneM2M"},
    "saref4env":           {"gt_dir": "saref4env",           "onto_file": "saref4env.ttl",           "label": "SAREF4ENV"},
    "videogameontology":   {"gt_dir": "videogameontology",   "onto_file": "videogameontology.owl",   "label": "VGO"},
    "vicinitycore":        {"gt_dir": "vicinitycore",        "onto_file": "vicinitycore.ttl",        "label": "VC"},
}

LLMS = ["qwen-max", "gpt-5", "glm-5"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_ground_truth(dataset_key: str) -> list[str]:
    info = DATASETS[dataset_key]
    path = os.path.join(DATASET_BASE, info["gt_dir"], f"{info['gt_dir']}.json")
    with open(path) as f:
        return json.load(f)["competency_questions"]


def load_generated_cqs(dataset_key: str) -> list[dict]:
    """Load generated CQs from all 3 LLMs and return a flat list with source info."""
    all_cqs = []
    for llm in LLMS:
        result_dir = os.path.join(RESULTS_BASE, dataset_key, "OntologyAgent", llm)
        if not os.path.isdir(result_dir):
            continue
        for fname in os.listdir(result_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            for cq in data["generated_cqs"]:
                all_cqs.append({"cq": cq, "llm": llm, "run": fname})
    return all_cqs


def strip_prefix(cq: str) -> str:
    """Remove [Simple]/[Intermediate]/[Complex] prefix."""
    for tag in ["[Simple] ", "[Intermediate] ", "[Complex] "]:
        if cq.startswith(tag):
            return cq[len(tag):]
    return cq


def extract_concepts(dataset_key: str) -> str:
    """Extract class and property names from ontology, return as compact string."""
    info = DATASETS[dataset_key]
    fpath = os.path.join(DATASET_BASE, info["gt_dir"], info["onto_file"])
    g = RDFGraph()
    fmt = "turtle" if fpath.endswith(".ttl") else "xml"
    g.parse(fpath, format=fmt)

    ns = g.namespace_manager
    classes = sorted({s.n3(namespace_manager=ns) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)})
    oprops = sorted({s.n3(namespace_manager=ns) for s in g.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)})
    dprops = sorted({s.n3(namespace_manager=ns) for s in g.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)})

    # Clean up prefixes for readability
    def clean(name):
        if ":" in name:
            return name.split(":", 1)[1]
        return name

    parts = []
    if classes:
        parts.append("Classes: " + ", ".join(clean(c) for c in classes))
    if oprops:
        parts.append("Properties: " + ", ".join(clean(p) for p in oprops + dprops))
    return "; ".join(parts)


def compute_matches(generated: list[str], ground_truth: list[str], model, threshold: float):
    """Return (matched_indices, unmatched_indices) based on cosine similarity."""
    pred_emb = model.encode(generated, convert_to_tensor=True)
    gt_emb = model.encode(ground_truth, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(pred_emb, gt_emb)

    matched = set()
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            if scores[i][j] > threshold:
                matched.add(i)
                break  # one match is enough

    unmatched = set(range(len(generated))) - matched
    return matched, unmatched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sample CQs for human evaluation")
    parser.add_argument("--n_unmatched", type=int, default=20, help="Unmatched CQs per dataset")
    parser.add_argument("--n_matched", type=int, default=10, help="Matched CQs per dataset (control)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine similarity threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print("Loading Sentence-BERT model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    all_rows = []

    for ds_key, ds_info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing {ds_info['label']} ({ds_key})")
        print(f"{'='*60}")

        # Load data
        gt_cqs = load_ground_truth(ds_key)
        gen_entries = load_generated_cqs(ds_key)

        # Deduplicate by cleaned CQ text
        seen = set()
        unique_entries = []
        for entry in gen_entries:
            clean_cq = strip_prefix(entry["cq"])
            if clean_cq not in seen:
                seen.add(clean_cq)
                unique_entries.append({**entry, "cq_clean": clean_cq})

        print(f"  Ground truth CQs: {len(gt_cqs)}")
        print(f"  Generated CQs (all LLMs, raw): {len(gen_entries)}")
        print(f"  Generated CQs (deduplicated): {len(unique_entries)}")

        # Compute matches
        clean_cqs = [e["cq_clean"] for e in unique_entries]
        matched_idx, unmatched_idx = compute_matches(clean_cqs, gt_cqs, sbert, args.threshold)

        print(f"  Matched (TP): {len(matched_idx)}")
        print(f"  Unmatched (FP): {len(unmatched_idx)}")

        # Sample
        matched_list = sorted(matched_idx)
        unmatched_list = sorted(unmatched_idx)
        random.shuffle(matched_list)
        random.shuffle(unmatched_list)

        sampled_matched = matched_list[:args.n_matched]
        sampled_unmatched = unmatched_list[:args.n_unmatched]

        print(f"  Sampled: {len(sampled_matched)} matched + {len(sampled_unmatched)} unmatched")

        # Extract ontology concepts for reference
        concepts = extract_concepts(ds_key)

        # Build rows (don't reveal matched/unmatched to annotators)
        for idx in sampled_matched + sampled_unmatched:
            entry = unique_entries[idx]
            all_rows.append({
                "dataset": ds_info["label"],
                "cq": entry["cq_clean"],
                "ontology_concepts": concepts,
                "source_llm": entry["llm"],
                # Hidden from annotators, for later analysis
                "_match_type": "matched" if idx in matched_idx else "unmatched",
            })

    # Shuffle all rows to prevent annotator bias
    random.shuffle(all_rows)

    # Assign IDs after shuffling
    for i, row in enumerate(all_rows, 1):
        row["id"] = i

    # ---------------------------------------------------------------------------
    # Write annotator CSV (without _match_type)
    # ---------------------------------------------------------------------------
    out_dir = os.path.join(PROJECT_ROOT, "human_eval")
    os.makedirs(out_dir, exist_ok=True)

    annotator_csv = os.path.join(out_dir, "annotation_sheet.csv")
    with open(annotator_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Dataset", "Competency Question", "Ontology Concepts",
                         "Fluency (Y/N)", "Relevance (Y/N)", "Answerability (Y/N)"])
        for row in all_rows:
            writer.writerow([row["id"], row["dataset"], row["cq"],
                             row["ontology_concepts"], "", "", ""])

    # ---------------------------------------------------------------------------
    # Write answer key (with _match_type for later analysis)
    # ---------------------------------------------------------------------------
    key_csv = os.path.join(out_dir, "answer_key.csv")
    with open(key_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Dataset", "Competency Question", "Match Type", "Source LLM"])
        for row in all_rows:
            writer.writerow([row["id"], row["dataset"], row["cq"],
                             row["_match_type"], row["source_llm"]])

    # ---------------------------------------------------------------------------
    # Write annotation guidelines
    # ---------------------------------------------------------------------------
    guidelines_path = os.path.join(out_dir, "annotation_guidelines.txt")
    with open(guidelines_path, "w", encoding="utf-8") as f:
        f.write(ANNOTATION_GUIDELINES)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Total CQs sampled: {len(all_rows)}")

    # Count per dataset and type
    stats = defaultdict(lambda: {"matched": 0, "unmatched": 0})
    for row in all_rows:
        stats[row["dataset"]][row["_match_type"]] += 1
    for ds, counts in sorted(stats.items()):
        print(f"  {ds}: {counts['matched']} matched + {counts['unmatched']} unmatched")

    print(f"\nOutput files:")
    print(f"  Annotation sheet : {annotator_csv}")
    print(f"  Answer key       : {key_csv}")
    print(f"  Guidelines       : {guidelines_path}")


# ---------------------------------------------------------------------------
# Annotation guidelines text
# ---------------------------------------------------------------------------
ANNOTATION_GUIDELINES = """\
================================================================================
         ANNOTATION GUIDELINES — CQGen-MAS Human Evaluation
================================================================================

TASK
----
You will evaluate a set of automatically generated Competency Questions (CQs)
for ontologies. Each row contains a CQ and the ontology it was generated for.

For each CQ, judge THREE dimensions by marking Y (Yes) or N (No):

1. FLUENCY
   Is the CQ a grammatically correct, understandable natural language question?
   - Y: The question reads naturally and is easy to understand.
   - N: The question has grammatical errors, is incoherent, or is not a question.

2. RELEVANCE
   Is the CQ thematically related to the ontology's conceptual domain?
   - Y: The question asks about concepts, entities, or relationships that fall
        within the ontology's scope (refer to the "Ontology Concepts" column).
   - N: The question asks about topics outside the ontology's domain.

3. ANSWERABILITY
   Could this CQ, in principle, be answered by querying the ontology's schema
   (classes, properties, and their relationships)?
   - Y: The question can be answered using the ontology's structure
        (even if no instance data is loaded).
   - N: The question requires external knowledge, instance data not derivable
        from the schema, or asks about concepts not present in the ontology.

IMPORTANT NOTES
---------------
- Judge each CQ independently. Do not compare CQs with each other.
- Use the "Ontology Concepts" column as reference for what the ontology covers.
- If you are unsure, lean toward Y (benefit of the doubt).
- Do NOT discuss your annotations with other annotators.

EXAMPLES
--------
Ontology: Video Game Ontology (VGO)
Concepts: Player, Achievement, Game, Character, Session, ...

  CQ: "Which achievements do players who own a specific character commonly have?"
  -> Fluency: Y | Relevance: Y | Answerability: Y

  CQ: "What percentage of players prefer action games?"
  -> Fluency: Y | Relevance: Y | Answerability: N
     (Statistical/distributional question cannot be answered from schema alone)

  CQ: "What is the weather forecast for tomorrow?"
  -> Fluency: Y | Relevance: N | Answerability: N

================================================================================
"""


if __name__ == "__main__":
    main()
