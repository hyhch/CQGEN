import os
import re
import json
import random
import metis
import asyncio
from rdflib import Graph, URIRef
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.logs import logger

# Louvain community detection (NetworkX)
import networkx as nx
from networkx.algorithms.community import louvain_communities

def louvain_partition(adj, resolution=1.0):
    G = nx.Graph()
    for u, neighbors in enumerate(adj):
        for v in neighbors:
            G.add_edge(u, v)
    comms = louvain_communities(G, weight="weight", resolution=resolution)
    node_to_comm = [-1 for _ in range(len(adj))]
    for cid, c in enumerate(comms):
        for u in c:
            node_to_comm[u] = cid
    return node_to_comm

# Leiden community detection (igraph + leidenalg)
import igraph as ig
import leidenalg as la

def leiden_partition(adjacency, resolution=1.0):
    seen = set()
    edges = []
    for u, nbrs in enumerate(adjacency):
        for v in nbrs:
            if u == v:
                continue  # skip self-loops
            e = (u, v) if u < v else (v, u)  # undirected dedup
            if e not in seen:
                seen.add(e)
                edges.append(e)
    g = ig.Graph(n=len(adjacency), edges=edges, directed=False)
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=resolution)
    return part.membership

# Spectral clustering (scikit-learn)
import numpy as np
from sklearn.cluster import SpectralClustering

def spectral_partition(adjacency, k):
    n = len(adjacency)
    adj_matrix = np.zeros((n, n), dtype=float)
    for u, nbrs in enumerate(adjacency):
        for v in nbrs:
            if u == v:
                continue
            adj_matrix[u, v] = adj_matrix[v, u] = 1
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans")
    labels = sc.fit_predict(adj_matrix)
    return labels


class LoadOWL(Action):
    """
    Parse OWL/RDF/TTL ontology files, extract namespaces and triples,
    return file path and parsed result dict.
    """
    name: str = "LoadOWL"

    async def run(self, file_path: str) -> dict:
        g = Graph()
        if file_path.endswith(".owl") or file_path.endswith(".rdf"):
            g.parse(file_path, format='xml')
        elif file_path.endswith(".ttl"):
            g.parse(file_path, format='turtle')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        namespaces = []
        triples = []
        for prefix, url in g.namespaces():
            namespaces.append({"prefix": prefix, "url": url})
        for s, p, o in g:
            if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
                triples.append(
                    (
                        str(s.n3(namespace_manager=g.namespace_manager)),
                        str(p.n3(namespace_manager=g.namespace_manager)),
                        str(o.n3(namespace_manager=g.namespace_manager))
                    )
                )
        return {"namespaces": namespaces, "triples": triples}


# ---------------------------------------------------------------------------
# Toolset definition for the segmentation algorithms (used by the LLM prompt)
# ---------------------------------------------------------------------------
SEGMENTATION_TOOLSET = json.dumps([
    {
        "name": "metis",
        "description": "METIS balanced k-way graph partitioning. Minimizes edge-cut "
                       "while producing even-sized subgraphs. Suitable for large, dense "
                       "graphs where balanced partition sizes are important.",
        "parameters": {
            "nparts": {
                "type": "integer",
                "description": "Number of partitions to create"
            }
        }
    },
    {
        "name": "louvain",
        "description": "Louvain community detection via greedy modularity optimization. "
                       "Discovers natural community structure. Suitable for graphs with "
                       "clear modular structure (high modularity coefficient).",
        "parameters": {
            "resolution": {
                "type": "float",
                "description": "Resolution parameter controlling community granularity "
                               "(higher = more, smaller communities)",
                "default": 1.0
            }
        }
    },
    {
        "name": "leiden",
        "description": "Leiden community detection, an improved Louvain variant "
                       "guaranteeing well-connected communities. Suitable for graphs "
                       "with high modularity where community connectivity matters.",
        "parameters": {
            "resolution": {
                "type": "float",
                "description": "Resolution parameter controlling community granularity "
                               "(higher = more, smaller communities)",
                "default": 1.0
            }
        }
    },
    {
        "name": "spectral",
        "description": "Spectral clustering via eigenvalues of the graph Laplacian. "
                       "Suitable for graphs with irregular or unclear community "
                       "structure and low modularity.",
        "parameters": {
            "k": {
                "type": "integer",
                "description": "Number of clusters to create"
            }
        }
    }
], indent=2)


class SelectSegmentationTool(Action):
    """
    Analyze the ontology graph's structural characteristics and use LLM to
    autonomously select the most appropriate segmentation algorithm.
    Corresponds to Listing 1 in the CQGen-MAS paper.
    """
    name: str = "SelectSegmentationTool"

    PROMPT_TEMPLATE: str = (
        "You are an agent in ontology engineering. Your task is to select the most "
        "appropriate partitioning algorithm and parameters to divide a (possibly large) "
        "ontology into semantically coherent subgraphs, such that downstream agents can "
        "operate within LLM context length limits.\n\n"
        "# Guidelines\n"
        "1. Analyze the structural characteristics of the ontology.\n"
        "2. Select which partitioning algorithm is most suitable and decide its parameters.\n"
        "3. Output strictly in the function-call format:\n"
        "```json\n"
        "{{\n"
        '  "tool_call": {{\n'
        '    "name": "<algorithm_name>",\n'
        '    "args": {{ /* parameters matching the tool schema */ }}\n'
        "  }}\n"
        "}}\n"
        "```\n\n"
        "# Accessible Toolset\n"
        "{toolset_definition}\n\n"
        "# Input Data\n"
        "- Number of nodes: {num_nodes}\n"
        "- Number of edges: {num_edges}\n"
        "- Average node degree: {avg_degree}\n"
        "- Modularity coefficient: {modular_coef}\n"
    )

    async def run(self, ontology: dict, max_graph_triples_num: int) -> dict:
        """Compute structural characteristics and ask the LLM to pick a tool."""
        triples = ontology["triples"]

        # --- Compute structural characteristics ---
        node_set = set()
        for s, _, o in triples:
            node_set.add(s)
            node_set.add(o)
        num_nodes = len(node_set)
        num_edges = len(triples)

        # If the graph is small enough to fit in one subgraph, skip selection
        if num_nodes <= max_graph_triples_num:
            logger.info(
                "Graph too small for segmentation (%d nodes); skipping tool selection.",
                num_nodes,
            )
            return {"method": "metis", "args": {}}

        avg_degree = round(2 * num_edges / num_nodes, 2) if num_nodes > 0 else 0

        # Compute modularity coefficient via a quick Louvain pass
        G = nx.Graph()
        for s, _, o in triples:
            G.add_edge(s, o)
        comms = louvain_communities(G, resolution=1.0)
        modularity_coef = round(nx.community.modularity(G, comms), 4)

        logger.info(
            "Structural characteristics: nodes=%d, edges=%d, avg_degree=%.2f, modularity=%.4f",
            num_nodes, num_edges, avg_degree, modularity_coef,
        )

        # --- Build prompt and ask LLM ---
        prompt = self.PROMPT_TEMPLATE.format(
            toolset_definition=SEGMENTATION_TOOLSET,
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            modular_coef=modularity_coef,
        )
        rsp = await self._aask(prompt)

        # --- Parse the function-call response ---
        selection = self._parse_tool_call(rsp, num_edges, max_graph_triples_num)
        logger.info(
            "LLM selected segmentation algorithm: %s with args %s",
            selection["method"], selection["args"],
        )
        return selection

    def _parse_tool_call(
        self, response: str, num_edges: int, max_graph_triples_num: int
    ) -> dict:
        """Extract the tool_call JSON from the LLM response."""
        default_nparts = max(2, num_edges // max_graph_triples_num)
        fallback = {"method": "metis", "args": {"nparts": default_nparts}}

        # Try ```json ... ``` block first
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to find a raw JSON object
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = response[start : end + 1]
            else:
                logger.warning(
                    "Could not extract JSON from LLM response; falling back to metis"
                )
                return fallback

        try:
            parsed = json.loads(json_str)
            tool_call = parsed.get("tool_call", parsed)
            method = tool_call.get("name", "metis")
            args = tool_call.get("args", {})

            valid_methods = {"metis", "louvain", "leiden", "spectral"}
            if method not in valid_methods:
                logger.warning(
                    "LLM selected unknown method '%s'; falling back to metis", method
                )
                return fallback

            return {"method": method, "args": args}
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(
                "Failed to parse LLM tool_call response (%s); falling back to metis", e
            )
            return fallback


class SegmentOntology(Action):
    """
    Segment parsed ontology triples into multiple subgraphs using a configurable
    graph partitioning algorithm.
    """
    name: str = "SegmentOntology"

    async def run(
        self,
        ontology: dict,
        max_graph_triples_num: int = 25,
        method: str = "metis",
        random_seed: int = 42,
        tool_args: dict | None = None,
    ) -> list:
        triples = ontology["triples"]
        namespaces = ontology["namespaces"]

        # Create mappings from nodes to unique integer IDs
        node_to_id = {}
        id_to_node = {}
        node_counter = 0
        for subj, _, obj in triples:
            for node in (subj, obj):
                if node not in node_to_id:
                    node_to_id[node] = node_counter
                    id_to_node[node_counter] = node
                    node_counter += 1

        # If the number of nodes fits within one subgraph, no segmentation needed
        if node_counter <= max_graph_triples_num:
            logger.info("Node count (%d) <= max_graph_triples_num (%d), skipping segmentation",
                        node_counter, max_graph_triples_num)
            return [ontology]

        # Create adjacency list
        adjacency = [[] for _ in range(node_counter)]
        for subj, _, obj in triples:
            subj_id = node_to_id[subj]
            obj_id = node_to_id[obj]
            adjacency[subj_id].append(obj_id)
            adjacency[obj_id].append(subj_id)

        # Resolve the actual segmentation method
        actual_method = method
        if method == "random":
            actual_method = random.Random(random_seed).choice(
                ["metis", "louvain", "leiden", "spectral"]
            )
        logger.info("Segmentation method: '%s' (requested: '%s', seed: %d)",
                     actual_method, method, random_seed)

        default_nparts = max(2, len(triples) // max_graph_triples_num)
        _args = tool_args or {}

        if actual_method == "metis":
            nparts = _args.get("nparts", default_nparts)
            metis_graph = metis.adjlist_to_metis(adjacency)
            _, parts = metis.part_graph(metis_graph, nparts=nparts)
        elif actual_method == "louvain":
            resolution = _args.get("resolution", 1.0)
            parts = louvain_partition(adjacency, resolution=resolution)
        elif actual_method == "leiden":
            resolution = _args.get("resolution", 1.0)
            parts = leiden_partition(adjacency, resolution=resolution)
        elif actual_method == "spectral":
            k = _args.get("k", default_nparts)
            parts = spectral_partition(adjacency, k=k)
        else:
            raise ValueError(f"Unknown segmentation method: {actual_method}")

        # Assign triples to partitions (use actual partition IDs from the algorithm)
        unique_parts = set(parts)
        subgraphs = {pid: [] for pid in unique_parts}
        for subj, pred, obj in triples:
            subj_part = parts[node_to_id[subj]]
            subgraphs[subj_part].append((subj, pred, obj))

        # Extract corresponding namespaces for each subgraph and return output
        subgraph_info_list = []
        subgraph_id_counter = 1
        for triples_chunk in subgraphs.values():
            # Skip empty subgraphs
            if not triples_chunk:
                logger.info("Skipping empty subgraph partition")
                continue

            subgraph_namespaces = []
            for triple in triples_chunk:
                for node in triple:
                    ns = self._get_namespace(node, namespaces)
                    if (ns["prefix"] or ns["url"]) and ns not in subgraph_namespaces:
                        subgraph_namespaces.append(ns)

            subgraph_info = {
                "subgraph_id": subgraph_id_counter,
                "namespaces": subgraph_namespaces,
                "triples": triples_chunk,
            }
            subgraph_info_list.append(subgraph_info)
            subgraph_id_counter += 1

        logger.info("Segmented ontology into %d non-empty subgraphs", len(subgraph_info_list))
        return subgraph_info_list

    def _get_namespace(self, node: str, namespaces: list) -> dict:
        if ":" in node:
            if node.startswith(":"):
                prefix = ""
            else:
                match = re.match(r"([^:]+):(.+)", node)
                if match:
                    prefix = match.group(1)
            for ns in namespaces:
                if ns["prefix"] == prefix:
                    return ns
        return {"prefix": "", "url": ""}


class OntologySegmenter(Role):
    name: str = "Oliver"
    profile: str = "OntologySegmenter"

    def __init__(
        self,
        max_graph_triples_num: int = 25,
        segmentation_method: str = "auto",
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_graph_triples_num = max_graph_triples_num
        self.segmentation_method = segmentation_method
        self.random_seed = random_seed
        self.selected_tool = None

        if segmentation_method == "auto":
            self.set_actions([LoadOWL, SelectSegmentationTool, SegmentOntology])
        else:
            self.set_actions([LoadOWL, SegmentOntology])
        self._set_react_mode(react_mode="by_order")

    def set_file_path(self, file_path: str) -> None:
        self.file_path = file_path

    def get_ontology_dict(self) -> dict:
        return self.ontology_dict

    def get_segmented_chunks_list(self) -> list:
        return self.segmented_chunks_list

    def get_selected_tool(self) -> dict | None:
        """Return the LLM-selected tool info (only available in 'auto' mode)."""
        return self.selected_tool

    async def _act(self) -> None:
        logger.info(f"{self._setting}: executing {self.rc.todo}")
        todo = self.rc.todo

        if isinstance(todo, LoadOWL):
            self.ontology_dict = await todo.run(self.file_path)
        elif isinstance(todo, SelectSegmentationTool):
            self.selected_tool = await todo.run(
                self.ontology_dict,
                self.max_graph_triples_num,
            )
        elif isinstance(todo, SegmentOntology):
            if self.selected_tool is not None:
                # "auto" mode: use LLM-selected method and args
                method = self.selected_tool["method"]
                tool_args = self.selected_tool["args"]
            else:
                # Manual mode: use the configured method
                method = self.segmentation_method
                tool_args = None
            self.segmented_chunks_list = await todo.run(
                self.ontology_dict,
                self.max_graph_triples_num,
                method=method,
                random_seed=self.random_seed,
                tool_args=tool_args,
            )
        else:
            raise ValueError(f"Unsupported action: {todo}")


if __name__ == "__main__":
    from config.Config import DATA_DIR_PATH, ONTOLOGY_NAME, SEGMENTATION_METHOD, RANDOM_SEED, MAX_NUM_SUB_TRIPLES

    async def main():
        json_path = os.path.join(DATA_DIR_PATH, ONTOLOGY_NAME, f"{ONTOLOGY_NAME}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            ontology_info = json.load(f)
        file_path = os.path.join(DATA_DIR_PATH, ONTOLOGY_NAME, ontology_info["file_name"])

        role = OntologySegmenter(
            max_graph_triples_num=MAX_NUM_SUB_TRIPLES,
            segmentation_method=SEGMENTATION_METHOD,
            random_seed=RANDOM_SEED,
        )
        role.set_file_path(file_path)
        logger.info("Processing ontology file: %s", file_path)
        await role.run("parse owl file into triples chunks")
        chunks = role.get_segmented_chunks_list()
        logger.info("Number of subgraph chunks: %d", len(chunks))

        if role.get_selected_tool():
            logger.info("LLM-selected tool: %s", role.get_selected_tool())

    asyncio.run(main())
