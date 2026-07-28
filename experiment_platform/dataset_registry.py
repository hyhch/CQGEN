"""Unified dataset mapping across all three CQ generation methods."""

import json
import os

# Root of the workspace
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to each project root
_ONTOLOGY_AGENT_ROOT = os.path.join(_WORKSPACE, "OntologyAgent")
_LLM4KE_ROOT = os.path.join(_WORKSPACE, "llm4ke")
_RETROFIT_ROOT = os.path.join(_WORKSPACE, "RETROFIT-CQs")

# Dataset mapping: name -> per-method info
DATASETS = {
    "demcare": {
        "description": "DemCare - Dementia care ontology",
        "ontology_agent": {
            "data_dir": os.path.join(_ONTOLOGY_AGENT_ROOT, "dataset"),
            "ontology_name": "demcare",
        },
        "llm4ke": {
            "data_dir": os.path.join(_LLM4KE_ROOT, "data"),
            "ontology_name": "DemCare",
        },
        "retrofit": {
            "ontology_path": os.path.join(
                _RETROFIT_ROOT, "Data", "Ontologies", "lov_demlab.rdf"
            ),
        },
    },
    "onem2m": {
        "description": "OneM2M - IoT base ontology",
        "ontology_agent": {
            "data_dir": os.path.join(_ONTOLOGY_AGENT_ROOT, "dataset"),
            "ontology_name": "onem2m",
        },
        "llm4ke": {
            "data_dir": os.path.join(_LLM4KE_ROOT, "data"),
            "ontology_name": "OneM2M",
        },
        "retrofit": {
            "ontology_path": os.path.join(
                _RETROFIT_ROOT, "Data", "Ontologies", "base_ontology.owl"
            ),
        },
    },
    "saref4env": {
        "description": "SAREF4ENV - Smart environment ontology",
        "ontology_agent": {
            "data_dir": os.path.join(_ONTOLOGY_AGENT_ROOT, "dataset"),
            "ontology_name": "saref4env",
        },
        "llm4ke": {
            "data_dir": os.path.join(_LLM4KE_ROOT, "data"),
            "ontology_name": "saref4env",
        },
        "retrofit": {
            "ontology_path": os.path.join(
                _RETROFIT_ROOT, "Data", "Ontologies", "saref4env.ttl"
            ),
        },
    },
    "vicinitycore": {
        "description": "VICINITY Core - IoT interoperability ontology",
        "ontology_agent": {
            "data_dir": os.path.join(_ONTOLOGY_AGENT_ROOT, "dataset"),
            "ontology_name": "vicinitycore",
        },
        "llm4ke": {
            "data_dir": os.path.join(_LLM4KE_ROOT, "data"),
            "ontology_name": "vicinitycore",
        },
        "retrofit": {
            "ontology_path": os.path.join(
                _RETROFIT_ROOT, "Data", "Ontologies", "vicinitycore.owl"
            ),
        },
    },
    "videogameontology": {
        "description": "Video Game Ontology",
        "ontology_agent": {
            "data_dir": os.path.join(_ONTOLOGY_AGENT_ROOT, "dataset"),
            "ontology_name": "videogameontology",
        },
        "llm4ke": {
            "data_dir": os.path.join(_LLM4KE_ROOT, "data"),
            "ontology_name": "videogameontology",
        },
        "retrofit": {
            "ontology_path": os.path.join(
                _RETROFIT_ROOT, "Data", "Ontologies", "videogameontology.owl"
            ),
        },
    },
}


def get_dataset_names():
    """Return list of available dataset names."""
    return list(DATASETS.keys())


def get_dataset_info(name):
    """Return dataset info dict for the given dataset name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


def load_ground_truth(dataset_name):
    """Load ground truth CQs from OntologyAgent's JSON files.

    Returns:
        List of ground truth CQ strings.
    """
    info = get_dataset_info(dataset_name)
    oa_info = info["ontology_agent"]
    json_path = os.path.join(
        oa_info["data_dir"], oa_info["ontology_name"],
        f"{oa_info['ontology_name']}.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("competency_questions", [])


def load_ground_truth_labels(dataset_name):
    """Load Simple/Complex labels for ground truth CQs.

    Returns:
        Dict mapping CQ string -> label ("Simple" or "Complex"),
        or None if no label file exists for this dataset.
    """
    info = get_dataset_info(dataset_name)
    oa_info = info["ontology_agent"]
    label_path = os.path.join(
        oa_info["data_dir"], oa_info["ontology_name"],
        f"{oa_info['ontology_name']}_label.json"
    )
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["question"].strip(): item["label"] for item in data}
