import os
import json
import rdflib
from config.Config import DATA_DIR_PATH

for ontology_name in os.listdir(DATA_DIR_PATH):
    # if ontology_name == "vicinitycore":
    #     continue
    ontology_json_path = os.path.join(DATA_DIR_PATH, ontology_name, ontology_name + ".json")
    with open(ontology_json_path, 'r', encoding='utf-8') as f:
        ontology_info = json.load(f)
    ontology_file_path = os.path.join(DATA_DIR_PATH, ontology_name, ontology_info["file_name"])
    g = rdflib.Graph()
    if ontology_file_path.endswith(".ttl"):
        g.parse(ontology_file_path, format="turtle")
    elif ontology_file_path.endswith(".owl") or ontology_file_path.endswith(".rdf"):
        g.parse(ontology_file_path, format="xml")
    else:
        raise ValueError(f"Unsupported file format: {ontology_file_path}")

    triples = []
    entities = set()
    for s, p, o in g:
        if isinstance(s, rdflib.URIRef) and isinstance(p, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            triples.append(
                (
                    str(s.n3(namespace_manager=g.namespace_manager)),
                    str(p.n3(namespace_manager=g.namespace_manager)),
                    str(o.n3(namespace_manager=g.namespace_manager))
                )
            )
            entities.add(str(s.n3(namespace_manager=g.namespace_manager)))
            entities.add(str(o.n3(namespace_manager=g.namespace_manager)))

    print(ontology_name, len(triples), len(entities), len(ontology_info["competency_questions"]))