"""Entity coverage metrics for CQ generation methods.

Provides:
- Entity extraction from OntologyAgent-format triples
- Entity name normalization (strip prefix, split camelCase)
- String-based coverage (for baselines: LLM4KE, RETROFIT-CQ)
- Retroactive coverage from existing OA results
- Per-iteration coverage averaging from OA run snapshots
- Entity loading for a given dataset
"""

import json
import os
import re

# Meta-entity prefixes to exclude from coverage computation
_META_PREFIXES = ("owl:", "rdf:", "rdfs:", "xsd:")

# Workspace root (parent of experiment_platform/)
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OA_ROOT = os.path.join(_WORKSPACE, "OntologyAgent")


def extract_entities_from_oa_triples(triples, exclude_meta=True):
    """Extract unique entities (subjects + objects) from OA-format triples.

    Args:
        triples: List of [subject, predicate, object] lists.
        exclude_meta: If True, exclude entities with owl:/rdf:/rdfs:/xsd: prefixes.

    Returns:
        Set of entity strings.
    """
    entities = set()
    for triple in triples:
        entities.add(triple[0])  # subject
        entities.add(triple[2])  # object
    if exclude_meta:
        entities = {
            e for e in entities
            if not any(e.startswith(p) for p in _META_PREFIXES)
        }
    return entities


def normalize_entity(entity):
    """Normalize an entity name for string matching.

    Steps:
    1. Strip namespace prefix (e.g., 'BO:OperationInput' -> 'OperationInput')
    2. Split camelCase into words (e.g., 'OperationInput' -> 'operation input')

    Returns:
        Lowercase normalized string.
    """
    # Strip namespace prefix
    if ":" in entity:
        entity = entity.split(":", 1)[1]
    # Strip URI fragment
    if "#" in entity:
        entity = entity.rsplit("#", 1)[1]
    if "/" in entity:
        entity = entity.rsplit("/", 1)[1]

    # Split camelCase: insert space before uppercase letters
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", entity)
    # Also split on underscores and hyphens
    words = re.sub(r"[_\-]", " ", words)
    return words.lower().strip()


def compute_string_coverage(entity_names, generated_cqs):
    """Compute entity coverage via case-insensitive substring matching.

    For each normalized entity, check if it appears in any generated CQ text.

    Args:
        entity_names: Set or list of raw entity strings (e.g., 'BO:OperationInput').
        generated_cqs: List of generated CQ strings.

    Returns:
        Dict with total_entities, covered_entities, coverage_rate, uncovered.
    """
    if not entity_names:
        return {
            "total_entities": 0,
            "covered_entities": 0,
            "coverage_rate": 0.0,
            "uncovered": [],
        }

    # Normalize all entities
    entity_list = list(entity_names)
    normalized = [normalize_entity(e) for e in entity_list]

    # Build lowercase CQ text corpus
    cq_texts_lower = [cq.lower() for cq in generated_cqs]

    covered = 0
    uncovered = []
    for raw, norm in zip(entity_list, normalized):
        if not norm:
            # Skip empty normalized names
            covered += 1
            continue
        found = any(norm in cq_text for cq_text in cq_texts_lower)
        if found:
            covered += 1
        else:
            uncovered.append(raw)

    total = len(entity_list)
    rate = (covered / total * 100) if total > 0 else 0.0

    return {
        "total_entities": total,
        "covered_entities": covered,
        "coverage_rate": round(rate, 2),
        "uncovered": uncovered,
    }


def compute_oa_coverage_from_results(results_data):
    """Compute final coverage from existing OA results using global dedup.

    Collects all entities across subgraphs (excluding meta prefixes),
    deduplicates globally, then computes covered vs total.

    Args:
        results_data: Dict loaded from an OA results JSON file
                      (must contain 'subgraph_chunks' with 'triples').

    Returns:
        Dict with overall coverage statistics.
    """
    chunks = results_data.get("subgraph_chunks", [])
    if not chunks:
        # Fallback: return pre-computed stats if no raw chunks available
        if results_data.get("overall_coverage_stats"):
            return results_data["overall_coverage_stats"]
        return {}

    global_entities = set()
    global_covered = set()
    subgraph_details = []

    for idx, chunk in enumerate(chunks):
        triples = chunk.get("triples", [])
        uncovered = set(chunk.get("uncovered_entities", []))
        chunk_entities = extract_entities_from_oa_triples(triples, exclude_meta=True)

        global_entities |= chunk_entities
        covered_in_chunk = chunk_entities - uncovered
        global_covered |= covered_in_chunk

        # Per-subgraph detail (still useful for debugging)
        rate = (len(covered_in_chunk) / len(chunk_entities) * 100) if chunk_entities else 0
        subgraph_details.append({
            "subgraph_id": idx,
            "total_entities": len(chunk_entities),
            "covered_entities": len(covered_in_chunk),
            "coverage_rate": round(rate, 2),
        })

    total = len(global_entities)
    covered = len(global_covered & global_entities)
    rate = (covered / total * 100) if total > 0 else 0.0

    return {
        "total_subgraphs": len(chunks),
        "total_entities": total,
        "total_covered_entities": covered,
        "total_uncovered_entities": total - covered,
        "overall_coverage_rate": round(rate, 2),
        "subgraph_coverage_details": subgraph_details,
    }


def compute_oa_iteration_coverage(results_data):
    """Compute per-iteration average coverage from OA results.

    For new results with 'iteration_coverage_history': return the stored data.
    For old results: only final coverage is available (reported as single iteration).

    Args:
        results_data: Dict loaded from an OA results JSON file.

    Returns:
        List of dicts, one per iteration, with avg_coverage_rate.
    """
    # Check for pre-computed iteration history
    history = results_data.get("iteration_coverage_history")
    if history:
        return history

    # Fallback for old results: compute final-only coverage as "iteration 0"
    coverage = compute_oa_coverage_from_results(results_data)
    if not coverage:
        return []

    return [{
        "iteration": 0,
        "avg_coverage_rate": coverage["overall_coverage_rate"],
        "note": "Only final coverage available (pre-iteration-tracking results)",
    }]


def load_entities_for_dataset(dataset_name):
    """Load the full entity set for a dataset.

    Uses OA's results files (which contain all parsed triples) as the canonical
    entity source. Tries multiple results files until one is found. Falls back
    to parsing the ontology JSON file for raw triples.

    Args:
        dataset_name: Name of the dataset (e.g., 'demcare').

    Returns:
        Set of entity strings (excluding meta-entities).
    """
    dataset_dir = os.path.join(_OA_ROOT, "dataset", dataset_name)

    # Try to load from any existing results file (they all contain the same triples)
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            fpath = os.path.join(dataset_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                triples = data.get("triples", [])
                if triples:
                    return extract_entities_from_oa_triples(triples, exclude_meta=True)
            except (json.JSONDecodeError, KeyError):
                continue

    # Fallback: load the dataset JSON (which may have triples if it was ever run)
    json_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        triples = data.get("triples", [])
        if triples:
            return extract_entities_from_oa_triples(triples, exclude_meta=True)

    # Last resort: parse OWL file with rdflib
    return _parse_entities_from_owl(dataset_name, dataset_dir)


def _parse_entities_from_owl(dataset_name, dataset_dir):
    """Parse entities directly from the OWL/RDF/TTL file using rdflib.

    Args:
        dataset_name: Name of the dataset.
        dataset_dir: Path to the dataset directory.

    Returns:
        Set of entity strings (local names only).
    """
    json_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    if not os.path.exists(json_path):
        return set()

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    owl_file = meta.get("file_name", "")
    owl_path = os.path.join(dataset_dir, owl_file)
    if not os.path.exists(owl_path):
        return set()

    try:
        import rdflib
        g = rdflib.Graph()
        ext = os.path.splitext(owl_path)[1].lower()
        fmt_map = {".owl": "xml", ".rdf": "xml", ".ttl": "turtle"}
        g.parse(owl_path, format=fmt_map.get(ext, "xml"))

        entities = set()
        for s, p, o in g:
            for term in (s, o):
                if isinstance(term, (rdflib.URIRef,)):
                    local = str(term).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                    if local and not any(local.startswith(mp.split(":")[0]) for mp in _META_PREFIXES):
                        entities.add(local)
        return entities
    except Exception:
        return set()
