import os
import re
import asyncio
import json
import copy
from rdflib import Graph, URIRef, Literal
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.logs import logger
from config.Config import MAX_EVAL_ITER

class SPARQLQuery(Action):
    """
    将能力问题转换为 SPARQL 查询并执行
    """
    name: str = "SPARQLQuery"
    GENERATE_PROMPT_TEMPLATE: str = """
    Generate SPARQL query for question: {question}, 
    based on the ontology triples: {triples}.
    
    ### Instructions for you to provide the query
    1. First, extract the paths and relations from the provided ontology triples by identifying relevant entities and predicates.
    2. Then generate SPARQL query using those paths and relations, ensuring the query aligns with the question's intent.
    3. Define all the necessary prefixes fully and use only those prefixes in the query.
    4. Use precise prefixes while generating a SPARQL query; do not give multiple options for prefixes in the query.
    5. Do not use '/' or '\\textbackslash' inside the WHERE clause; define the full path in prefix.
    6. Do not include any explanations or apologies in your responses.
    7. Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
    8.  Do not include any text except the SPARQL query generated.
    
    ### Notes
    Even if the ontology lacks individual instances (i.e., the query may return no results), the goal is still to generate a **syntactically and semantically correct** SPARQL query that reflects the competency question accurately.

    ### Output requirement
    Output must be in the following format, without any additional explanations or commentary:
    ```sql
    # Write your query here.
    ```

    ### Examples
    1.  - Question: "What is the gender information?"
        - SPARQL Query:
            ```sql
            SELECT ?x
            WHERE {{
                ?x rdf:type lab:GenderType
            }}
            ```
    2.  - Question: "What data are measured for interaction assessment?"
        - SPARQL Query:
            ```sql
            select distinct *
            where
            {{
                [] rdfs:subClassOf event:SocialInteraction, [
                owl:onProperty ?p;
                owl:someValuesFrom []
                ].
            }}
            ```
    3.  - Question: "Which stuffs have as part exactly two substuffs?"
        - SPARQL Query:
            ```sql
            select distinct *
            where
            {{
            ?stuff rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty :hasSubStuff ;
            owl:cardinality "2"^^xsd:nonNegativeInteger
            ]
            }}
            ```
    """

    CORRECT_PROMPT_TEMPLATE: str = """
    You are an expert in ontology and SPARQL. Your task is to correct the SPARQL query based on the feedback provided.

    Correct the following SPARQL query: 
    ```sql
    {sparql_query}
    ```
    for the question: {question}

    ## Error Analysis
    1. Parse the error message: {feedback}. Identify the root cause of the error in the SPARQL query.
    2. Analyze the ontology triples: {triples}. Check if the query aligns with the provided ontology triples. Focus on:
        - Any syntax errors, undefined prefixes, semantic inconsistencies?
        - If the intended meaning of the query logically aligns with the provided ontology triples?
        - Any missing prefix info?

    ## Correction Rules
    1. Preserve original query intent: Ensure that the modified query still answers the question as intended, without changing its core logic.
    2. Fix the bugs within the query, e.g., syntax errors, undefined prefixes, type conflicts, etc.
    3. Do not change query type (SELECT/ASK/CONSTRUCT).

    ## Outpur Requirements
    Output must be in the following format, without any additional explanations or commentary:
    ```sql
    # Write your query here.
    ```
    """

    RETHINK_PROMPT_TEMPLATE: str = """
    # Role & Task
    You are an expert in ontology modeling and SPARQL query optimization.
    Your task is to analyze why a given SPARQL query returned no results, and either fix the query or confirm that the empty result is logically valid based on the ontology.

    # Inputs
    - Question: {question}
    - Ontology Triples: {triples}
    - Current SPARQL Query (returned empty): {sparql_query}
    
    # Analysis Steps
    1. Terminology Check: Compare query terms with ontology terms to identify mismatches (e.g., singular/plural, synonyms).
    2. Structure Validation: Verify if query patterns (e.g., triple paths) exist in the ontology or have alternatives.
    3. Complexity Handling: Consider if FILTER/OPTIONAL clauses are overly restrictive or missing.
    4. True Emptiness Confirmation: Schema required classes/properties but no instances. In this case, the empty result is valid and no query modification is needed.

    # Correction Rules
    1. Fixable issues (output revised query):
        - Map query terms to ontology terms strictly.
        - Maintain correct logic and syntax aligned with the original intent of the question.
        ```sql
        # Write your query here.
        ```
    2. True emptiness (output Y if all met):
        - All query terms match ontology terms.
        - No syntax/logic errors in query.
        - Ontology supports required classes/properties.
        - The ontology triples simply lacks instance data, and the empty result is valid.
        ```sql
        Y
        ```

    # Output Requirements
    Output must be in the following format, without any additional explanations or commentary:
    ```sql
    # Write your query here.
    ```
    OR
    ```sql
    Y
    ```

    # Examples
    1. Valid empty case
        - Question: "What is the username of the player?"
        - Ontology Triples: [(ex:username, rdfs:domain, ex:Player), (ex:username, "rdf:type", "owl:DatatypeProperty"), (ex:Player, "rdf:type", "owl:Class")]
        - Current SPARQL Query (returned empty output): "SELECT ?username\n WHERE {{\n?player rdf:type ex:Player.\n?username rdfs:domain ?player.\n?username rdf:type ex:username.\n}}"
        - Output:
            ```sql
            Y
            ```

    2. Fixable case
        - Question: "Which cities are located in France?"
        - Ontology Triples: [(ex:City, rdf:type, owl:DatatypeProperty), (ex:Country, rdf:type, owl:Class), (ex:City, ex:locatedIn ex:Country)]
        - SPARQL Query: "SELECT ?city\nWHERE {{\n?city rdf:type ex:City.\n?city ex:locatedIn ex:France.\n}}"
        - Output:
            ```sql
            ask where
            {{
                ex:City ex:locatedIn ex:Country.
            }}
            ```
    """

    def extract_query(self, response: str) -> str:
        pattern = r"```sql(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        query = matches[-1].strip() if matches else response
        return query

    def rebuild_graph(self, namespaces: list, triples: list) -> Graph:
        g = Graph()

        for ns in namespaces:
            g.bind(ns["prefix"], ns["url"])

        for s, p, o in triples:
            s = self.restore_uri_or_literal(s, namespaces)
            p = self.restore_uri_or_literal(p, namespaces)
            o = self.restore_uri_or_literal(o, namespaces)
            g.add((s, p, o))
        return g

    def restore_uri_or_literal(self, value: str, namespaces: list):
        for ns in namespaces:
            if value.startswith(f"{ns['prefix']}:"):
                return URIRef(f"{ns['url']}{value.split(':', 1)[1]}")
        return Literal(value)

    def execute_sparql_query(self, sparql_query: str, namespaces: list, triples: list) -> list:
        g = self.rebuild_graph(namespaces, triples)
        # Execute the SPARQL query
        result = g.query(sparql_query)
        return [str(item.n3(namespace_manager=g.namespace_manager)) for row in result for item in row]

    async def run(self, chunk_info: dict) -> dict:
        namespaces = chunk_info["namespaces"]
        triples = chunk_info["triples"]
        competency_questions = chunk_info["competency_questions"]
        qualified_questions = copy.deepcopy(chunk_info.get("qualified_questions", []))
        for question in competency_questions:
            # check if the query is already qualified
            if any([question == q["question"] for q in qualified_questions]):
                continue
            # generate sparql query
            prompt = self.GENERATE_PROMPT_TEMPLATE.format(question=question, triples=triples)
            rsp = await self._aask(prompt)
            sparql_query = self.extract_query(rsp)
            is_qualified = False
            query_result = list()
            for iter_count in range(MAX_EVAL_ITER):
                logger.info(f"Iteration {iter_count} for question: {question}")
                # execute sparql query
                try:
                    query_result = self.execute_sparql_query(sparql_query, namespaces, triples)
                    is_qualified = True # 只要执行成功，就认为是qualified
                    # if got query result from the triples, break; otherwise rethink it
                    if len(query_result) > 0:
                        break
                    else:
                        prompt = self.RETHINK_PROMPT_TEMPLATE.format(question=question, sparql_query=sparql_query, triples=triples)
                        rsp = await self._aask(prompt)
                        rethink_query = self.extract_query(rsp)
                        if rethink_query.strip() == "Y":
                            # Keep the last valid SPARQL instead of overwriting
                            # with "Y" — entity names in the query text are
                            # needed by coverage checking downstream.
                            break
                        else:
                            sparql_query = rethink_query
                            continue
                except Exception as e:
                    is_qualified = False
                    feedback = str(e)
                    prompt = self.CORRECT_PROMPT_TEMPLATE.format(sparql_query=sparql_query, question=question, feedback=feedback, triples=triples)
                    rsp = await self._aask(prompt)
                    sparql_query = self.extract_query(rsp)
                    continue
            if is_qualified:
                qualified_questions.append(
                    {
                        "question": question,
                        "sparql_query": sparql_query,
                        "query_result": query_result
                    }
                )
        chunk_info["qualified_questions"] = qualified_questions
        return chunk_info

class ExtractUncoverEntity(Action):
    """
    更新能力问题 & 提取 uncovered entities
    """
    name: str = "CheckCQ"

    # OWL/RDF/RDFS vocabulary prefixes — these are structural meta-terms,
    # not domain concepts, so they should be excluded from coverage checks.
    _META_PREFIXES = ("owl:", "rdf:", "rdfs:", "xsd:")

    def check_cq(self, question: str, qualified_questions: list):
        for q in qualified_questions:
            if question == q["question"]:
                return True
        return False

    def _is_meta_entity(self, node: str) -> bool:
        """Return True for OWL/RDF/RDFS vocabulary terms (e.g. owl:Class)."""
        return any(node.startswith(p) for p in self._META_PREFIXES)

    def check_entity(self, node: str, qualified_questions: list):
        for q in qualified_questions:
            # Substring match in the SPARQL query text
            if node in q["sparql_query"]:
                return True
            # Substring match in any query result element
            # (handles n3 serialization differences, e.g. extra wrapping quotes)
            for result_item in q["query_result"]:
                if node in result_item or result_item in node:
                    return True
        return False

    async def run(self, chunk_info: dict) -> dict:
        # filter competency questions
        filtered_competency_questions = list()
        for question in chunk_info["competency_questions"]:
            if self.check_cq(question, chunk_info["qualified_questions"]):
                filtered_competency_questions.append(question)
        chunk_info["competency_questions"] = filtered_competency_questions

        # extract uncovered entities (excluding OWL/RDF/RDFS meta-terms)
        uncovered_entities = set()
        for triple in chunk_info["triples"]:
            subject = triple[0]
            if not self._is_meta_entity(subject) and \
               not self.check_entity(subject, chunk_info["qualified_questions"]):
                uncovered_entities.add(subject)
            object = triple[2]
            if not self._is_meta_entity(object) and \
               not self.check_entity(object, chunk_info["qualified_questions"]):
                uncovered_entities.add(object)
        chunk_info["uncovered_entities"] = list(uncovered_entities)
        return chunk_info

class SPARQLEvaluator(Role):
    name: str = "Edwards"
    profile: str = "SPARQLEvaluator"

    def __init__(self, **kwargs):
        """
        初始化 SPARQLEvaluator 角色
        """
        super().__init__(**kwargs)
        self.set_actions([SPARQLQuery, ExtractUncoverEntity])
        self._set_react_mode(react_mode="by_order")

    def set_chunk_info(self, chunk_info: dict) -> None:
        self.chunk_info = chunk_info
        return

    async def _act(self) -> None:
        """
        定义角色行动逻辑
        """
        logger.info(f"{self._setting}: executing {self.rc.todo}")
        todo = self.rc.todo
        self.chunk_info = await todo.run(self.chunk_info)
        return

    def get_chunk_info(self) -> dict:
        return self.chunk_info

async def main():
    from config.Config import DATA_DIR_PATH, ONTOLOGY_NAME
    input_json_file = os.path.join(DATA_DIR_PATH, ONTOLOGY_NAME, ONTOLOGY_NAME + ".json")
    with open(input_json_file, 'r', encoding='utf-8') as f:
        ontology_info = json.load(f)

    # Load ontology file and segment it into chunks
    from roles.ontology_segmenter import OntologySegmenter
    from roles.competency_question_generator import CQGenerator
    task = "parse owl file into triples chunks"
    segmenter = OntologySegmenter()
    ontology_file_path = os.path.join(DATA_DIR_PATH, ONTOLOGY_NAME, ontology_info["file_name"])
    segmenter.set_file_path(ontology_file_path)
    await segmenter.run(task)
    chunks_list = segmenter.get_segmented_chunks_list()

    generator = CQGenerator()
    evaluator = SPARQLEvaluator()
    for chunk_info in chunks_list:
        # Generate competency questions
        task = "generate competency questions"
        generator.set_input(chunk_info, ontology_info["description"], ontology_info["competency_questions"][:10])
        await generator.run(task)
        chunk_info = generator.get_chunk_info()
        logger.info(f"Generated [{len(chunk_info['competency_questions'])}] Competency Questions.")

        # Evaluate competency questions
        evaluator.set_chunk_info(chunk_info)
        await evaluator.run(task)
        chunk_info = evaluator.get_chunk_info()
        logger.info(f"Found {len(chunk_info['uncovered_entities'])} uncovered by " + \
                    f"evaluating {len(chunk_info['competency_questions'])} competency questions.")

if __name__ == "__main__":
    asyncio.run(main())