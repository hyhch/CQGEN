import re
import asyncio
import json
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.logs import logger

def decode_cq(questions: str) -> list:
    """
    Decodes a string containing questions into a list of formatted question strings.
    """
    pattern = re.compile(r"(\d+)\.\s*\[(Simple|Complex)\]\s*(.*?)\s*\n")
    questions_list = []
    for match in pattern.finditer(questions):
        difficulty = match.group(2)
        question_text = match.group(3).strip()
        questions_list.append(f"[{difficulty}] {question_text}")
    return questions_list

class WriteCQ(Action):
    """
    为输入的本体三元组片段生成能力问题
    """
    name: str = "WriteCQ"
    PROMPT_TEMPLATE: str = """
    # Role & Task
    You are an expert in ontology engineering. Your task is to generate as many competency questions as possible in different complexity levels based on the given ontology triples.

    # Definition of Competency Question
    Competency questions are the questions that outline the scope of ontology and provide an idea about the knowledge that needs to be entailed in the ontology.
    For example,
    1. For the ontology triple `[(ex:Person, ex:hasName, xsd:string)]`, competency questions could be:
        - "What is the name of a person?"
        - "How many people have the same name?"
    2. For the ontology triples `[(ex:Movie, ex:hasActor, ex:Person), (ex:Movie, ex:hasDirector, ex:Person), (ex:Movie, ex:hasGenre, ex:Genre), (ex:Person, ex:birthPlace, ex:Place)]`, competency questions could be:
        - "Who acted in the movie 'Pulp Fiction'?"
        - "Where is the actor born?"
        - "What are all the movies that he has acted in?"
        - "What genres exist in the ontology?"
        - "Which action movies were directed by people born in France?"
        - "Who acted in both comedy and drama movies?"
        - "Which actors were born in the same city as the director of 'Inception'?"

    # Question Complexity Guidelines
    1. [Simple]: Single-class/property retrieval and direct triple patterns requiring no reasoning. For examples,
        - "What is the definition of Mammal?" (concept definition)
        - "Who founded Apple Inc.?" (single fact retrieval)
        - "What are the symptoms of Diabetes?" (instance enumeration)

    2. [Complex]: Multi-hop queries (topological distance >= 2) that traverse multiple ontology classes, properties, or relations. May involve cross-relationship reasoning, conditional filtering, statistical operations, or aggregation. For examples,
        - "Which companies were founded by Stanford alumni?" (cross-relationship query)
        - "Which cities in California have populations over 1 million?" (multiple constraints)
        - "If a patient has these symptoms and is allergic to penicillin, what treatments are available?" (complex filtering)
        - "How would Policy A affect Industry B if implemented in 2025?" (temporal and causal reasoning)

    # Output Requirements:
    1. Each question must in everyday language (avoid technical terms from ontology like class name, property, SPARQL termilogy, etc.).
    2. Cover both complexity levels as defined above.
    3. Provide questions in the following format, without any additional explanations or commentary:
        ```questions
        1. [Simple] Question 1?
        2. [Complex] Question 2?
        ```

    # Related Infomation
    - The description of the given ontology: {description}
    - Example competency questions for the ontology:
    {example_questions}

    # Input
    Now, you are only given parts of the ontology triples, and you need to generate competency questions based on them, following the same format as above:
    {triples}.

    # Output only this block:
    ```questions
    # Your questions here
    ```
    """
    async def run(self, triples: list, description: str = "", example_questions: list = []) -> list:
        prompt = self.PROMPT_TEMPLATE.format(triples=triples, 
                                             description=description,
                                             example_questions=example_questions)
        rsp = await self._aask(prompt)
        competency_questions_list = decode_cq(rsp)
        return competency_questions_list

class ExpandCQ(Action):
    """
    根据输入的本体三元组片段、现有能力问题和未覆盖的实体扩展新的能力问题
    """
    name: str = "ExpandCQ"
    PROMPT_TEMPLATE: str = """
    You are an expert in ontology engineering. Your task is to generate additional simple competency questions to cover currently unaddressed entities in the ontology.

    Input:
    - Ontology Triples: {ontology}
    - Existing Questions: {existing_questions}
    - Uncovered Entities: {uncovered_entities}

    Generation Rules:
    1. Focus Requirement: Only generate questions that include at least one uncovered entity.
    2. Question Style: Use a natural, conversational tone. Avoid technical or formal terminology (e.g., ontology class names or SPARQL terms).
    3. Complexity Control: Only generate **Simple** questions, e.g. single class/property retrieval or direct triple patterns — no multi-hop reasoning or complex chaining.
    4. Uniqueness Check: Ensure no duplication with existing questions.

    Output Format:
    If you can generate valid questions for the uncovered entities, list them in the following format:
    ```questions
    1. [Simple] Question 1?
    2. [Simple] Question 2?
    3. [Simple] Question 3?
    ```
    If no suitable questions can be generated, return the following format with no added questions:
    ```questions
    ```
    """

    async def run(self, triples: list, competency_questions: list, uncovered_entities: list) -> list:
        prompt = self.PROMPT_TEMPLATE.format(ontology=triples, 
                                             existing_questions=competency_questions,
                                             uncovered_entities=uncovered_entities)
        rsp = await self._aask(prompt)
        competency_questions_list = decode_cq(rsp)
        competency_questions.extend(competency_questions_list)
        return competency_questions

class RefineCQ(Action):
    """
    根据本体片段，对输入的能力问题进行润色
    """
    name: str = "RefineCQ"
    PROMPT_TEMPLATE: str = """
    Your task is to refine these competency questions based on the provided ontology triples.
    Competency Questions: {questions}
    Ontology Triples: {triples}

    Follow these principles during refinement:
    1. Clarity: Use natural, conversational language and ensure the questions are clear, concise, and easy to understand. For example,
        - "What kinds of things can happen while you're playing a game?" is better than "What types of GameEvents occur in a video game?"
        - "Can you lose and gain items at the same time in a game?" is better than "Can a LoseEvent occur at the same time as a GainEvent in a video game?"
    2. Relevance Check: Make sure the questions are relevant to the given ontology. Remove questions about concepts not in the triples.
    3. Deduplication: Merge similar questions where possible to avoid repetition.

    Please output the refined questions in the following format:
    ```questions
    1. [Simple] Question 1?
    2. [Complex] Question 2?
    ```
    Do not include any additional information or explanations.
    """

    async def run(self, triples: list, competency_questions: list) -> list:
        prompt = self.PROMPT_TEMPLATE.format(triples=triples, questions=competency_questions)
        rsp = await self._aask(prompt)
        refined_competency_questions = decode_cq(rsp)
        return refined_competency_questions

class CQGenerator(Role):
    name: str = "Chris"
    profile: str = "CQGenerator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_input(self, chunk_info: dict, description: str = "", cq_examples: list = []) -> None:
        self.chunk_info = chunk_info
        self.description = description
        self.cq_examples = cq_examples

        if "uncovered_entities" not in chunk_info:
            self.set_actions([WriteCQ])
        else:
            self.set_actions([ExpandCQ])
        self._set_react_mode(react_mode="by_order")
        return

    async def _act(self) -> None:
        logger.info(f"{self._setting}: executing {self.rc.todo}")
        todo = self.rc.todo

        if isinstance(todo, WriteCQ):
            self.chunk_info["competency_questions"] = \
                await todo.run(self.chunk_info["triples"],
                               self.description,
                               self.cq_examples)
        elif isinstance(todo, ExpandCQ):
            self.chunk_info["competency_questions"] = \
                await todo.run(self.chunk_info["triples"], 
                               self.chunk_info["competency_questions"],
                               self.chunk_info["uncovered_entities"])
        elif isinstance(todo, RefineCQ):
            self.chunk_info["competency_questions"] = \
                await todo.run(self.chunk_info["triples"], 
                               self.chunk_info["competency_questions"])
        else:
            raise ValueError(f"Unsupported action: {todo}")
        return

    def get_chunk_info(self) -> dict:
        return self.chunk_info

async def main():
    import os
    from config.Config import DATA_DIR_PATH, ONTOLOGY_NAME
    input_json_file = os.path.join(DATA_DIR_PATH, ONTOLOGY_NAME, ONTOLOGY_NAME + ".json")
    with open(input_json_file, 'r', encoding='utf-8') as f:
        ontology_info = json.load(f)

    # Load ontology file and segment it into chunks
    from roles.ontology_segmenter import OntologySegmenter
    task = "parse owl file into triples chunks"
    role = OntologySegmenter()
    ontology_file_path = os.path.join(DATA_DIR_PATH, ONTOLOGY_NAME, ontology_info["file_name"])
    role.set_file_path(ontology_file_path)
    await role.run(task)
    chunks_list = role.get_segmented_chunks_list()

    # Generate competency questions
    role = CQGenerator()
    task = "generate competency questions"
    for chunk_info in chunks_list:
        role.set_input(chunk_info, ontology_info["description"], ontology_info["competency_questions"][:10])
        await role.run(task)
        chunk_info = role.get_chunk_info()
        logger.info(f"Generated [{len(chunk_info['competency_questions'])}] Competency Questions.")

        # 模拟所有节点都没生成，再进行扩展
        uncovered_entities = set()
        for triple in chunk_info["triples"]:
            uncovered_entities.add(triple[0])
            uncovered_entities.add(triple[2])
        chunk_info["uncovered_entities"] = list(uncovered_entities)

        role.set_input(chunk_info, ontology_info["description"], ontology_info["competency_questions"][:10])
        await role.run(task)
        chunk_info = role.get_chunk_info()
        logger.info(f"Extended to [{len(chunk_info['competency_questions'])}] Competency Questions.")
        logger.info("All CQs:\n" + "\n".join(chunk_info["competency_questions"]))
        break

if __name__ == "__main__":
    asyncio.run(main())