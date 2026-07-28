import argparse
import json
import logging
import os
from datetime import datetime


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="RETROFIT-CQs: Retrofit Competency Questions from ontologies using LLMs"
    )
    p.add_argument("--ontology", required=True, help="Path to ontology file (OWL/TTL/RDF)")
    p.add_argument("--llm", required=True, choices=["gpt4", "llama", "qwen"],
                    help="LLM backend to use")
    p.add_argument("--benchmark", default="Data/CQs/Only CQs.xlsx",
                    help="Path to benchmark CQs Excel file")
    p.add_argument("--benchmark-filter", default=None,
                    help="Ontology URI substring to filter benchmark CQs (auto-detected if omitted)")
    p.add_argument("--threshold", type=float, default=0.6,
                    help="SBERT cosine similarity threshold (default: 0.6)")
    p.add_argument("--use-deeponto", action="store_true",
                    help="Use DeepOnto instead of rdflib for triple extraction")
    p.add_argument("--skip-extract", action="store_true",
                    help="Skip triple extraction stage")
    p.add_argument("--skip-generate", action="store_true",
                    help="Skip CQ generation stage")
    p.add_argument("--triples-csv", default=None,
                    help="Path to existing triples CSV (use with --skip-extract)")
    p.add_argument("--generated-cqs", default=None,
                    help="Path to existing generated CQs Excel (use with --skip-generate)")
    p.add_argument("--output-dir", default="outputs",
                    help="Base directory for run outputs (default: outputs)")
    p.add_argument("--label", action="store_true",
                    help="Run optional CQ complexity labeling after evaluation")
    return p.parse_args(argv)


def detect_ontology_format(path):
    """Detect rdflib parse format from file extension."""
    ext = os.path.splitext(path)[1].lower()
    formats = {
        ".owl": "xml",
        ".rdf": "xml",
        ".ttl": "turtle",
        ".n3": "n3",
        ".nt": "ntriples",
        ".jsonld": "json-ld",
    }
    fmt = formats.get(ext, "xml")
    return fmt


def create_run_dir(base_dir="outputs"):
    """Create a timestamped run directory and return its path."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_logging(run_dir):
    """Configure logging to both console and run.log in run_dir."""
    log_path = os.path.join(run_dir, "run.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear any existing handlers
    logger.handlers.clear()
    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def save_run_config(run_dir, args):
    """Save run parameters as run_config.json."""
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path
