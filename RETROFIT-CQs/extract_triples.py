import csv
import logging
import re

from rdflib import Graph

logger = logging.getLogger(__name__)


def _extract_local_name(uri):
    """Extract the local name from a URI (after # or last /)."""
    s = str(uri)
    return s.split("#")[-1] if "#" in s else s.split("/")[-1]


def _is_valid(value):
    """Check if a value is a readable local name (letters/underscores, length > 2)."""
    return bool(re.match(r"^[a-zA-Z_]+$", value)) and len(value) > 2


def extract_triples_rdflib(ontology_path, output_csv, fmt="xml"):
    """Extract triples from an ontology file using rdflib.

    Args:
        ontology_path: Path to the ontology file.
        output_csv: Path to write the tab-delimited CSV output.
        fmt: rdflib parse format (xml, turtle, n3, etc.).

    Returns:
        Number of valid triples extracted.
    """
    logger.info("Loading ontology with rdflib: %s (format=%s)", ontology_path, fmt)
    g = Graph()
    g.parse(ontology_path, format=fmt)
    logger.info("Graph has %d raw statements", len(g))

    data = []
    for subj, pred, obj in g:
        sub = _extract_local_name(subj)
        pre = _extract_local_name(pred)
        ob = _extract_local_name(obj)
        if not sub or not pre or not ob:
            continue
        if not (_is_valid(sub) and _is_valid(pre) and _is_valid(ob)):
            continue
        data.append([sub, pre, ob])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)

    logger.info("Extracted %d valid triples -> %s", len(data), output_csv)
    return len(data)


def extract_triples_deeponto(ontology_path, output_csv):
    """Extract triples from an ontology file using DeepOnto.

    Args:
        ontology_path: Path to the ontology file.
        output_csv: Path to write the tab-delimited CSV output.

    Returns:
        Number of triples extracted.
    """
    logger.info("Loading ontology with DeepOnto: %s", ontology_path)
    from deeponto.onto import Ontology
    from deeponto.onto.projection import OntologyProjector

    onto = Ontology(ontology_path, ignore_failed_imports=True)
    projector = OntologyProjector(
        bidirectional_taxonomy=False, only_taxonomy=True, include_literals=True
    )
    projected = projector.project(onto)
    logger.info("Projected ontology has %d statements", len(projected))

    def _con(c):
        return str(c).rsplit("#")[-1]

    data = []
    for s, p, o in projected:
        data.append([_con(s), _con(p), _con(o)])

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)

    logger.info("Extracted %d triples -> %s", len(data), output_csv)
    return len(data)
