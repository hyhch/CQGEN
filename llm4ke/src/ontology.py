"""Ontology loading, schema extraction, and ground truth utilities."""

import os
import re
from pathlib import Path

import yaml
from rdflib import Graph, RDF, RDFS, OWL


def simplify(uri):
    """Strip namespace from URI, replace underscores with spaces."""
    return re.split(r'[#/]', str(uri))[-1].replace('_', ' ')


def load_ontology(input_path):
    """Load all OWL/RDF/TTL files from input_path/dm/ into an rdflib Graph."""
    g = Graph()
    dm_dir = os.path.join(input_path, 'dm')
    for filename in os.listdir(dm_dir):
        if filename.rsplit('.', 1)[-1] in ('rdf', 'ttl', 'owl'):
            g.parse(os.path.join(dm_dir, filename))
    return g


def extract_classes(graph):
    """Extract OWL class URIs from the graph."""
    return [s for s, _, _ in graph.triples((None, RDF.type, OWL.Class))]


def extract_properties(graph):
    """Extract OWL object and datatype property URIs from the graph."""
    return ([s for s, _, _ in graph.triples((None, RDF.type, OWL.ObjectProperty))] +
            [s for s, _, _ in graph.triples((None, RDF.type, OWL.DatatypeProperty))])


def extract_schema(graph, properties):
    """Build (domain, property, range) triples for properties with defined domains."""
    schema = []
    for p in properties:
        domain = graph.value(p, RDFS.domain, None)
        if domain is None:
            continue
        range_val = graph.value(p, RDFS.range, None)
        range_str = simplify(range_val) if range_val else 'literal'
        schema.append((simplify(domain), simplify(p), range_str))
    return schema


def load_ground_truth(input_path):
    """Load ground truth CQs from input_path/cqs/cqs.yml."""
    with open(os.path.join(input_path, 'cqs', 'cqs.yml')) as f:
        data = yaml.safe_load(f)
    return [c['question'] for c in data['ontology']['cqs']]


def load_description(input_path):
    """Load ontology description.txt if it exists, otherwise return empty string."""
    desc_path = os.path.join(input_path, 'description.txt')
    if os.path.exists(desc_path):
        return Path(desc_path).read_text()
    return ''


def select_in_batches(lst, batch_size=20):
    """Yield successive batches from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def props_for_classes(schema, class_names):
    """Get unique property names related to the given class names."""
    return list({p for s, p, o in schema if s in class_names or o in class_names})


def schema_for_classes(schema, class_names):
    """Get schema triples related to the given class names."""
    return [(s, p, o) for s, p, o in schema if s in class_names or o in class_names]
