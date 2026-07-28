import logging
import os
from abc import ABC, abstractmethod

import openai
import pandas as pd
from openpyxl import Workbook

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "As an ontology engineer, Provide competency questions focused on the context "
    "provided; avoid using narrative questions. competency questions are the questions "
    "that outline the scope of an ontology and provide an idea about the knowledge that "
    "needs to be entailed in the ontology.Please use 1. XXXX this format to generate CQ, "
    "and do not contain any other content"
)


# ---------------------------------------------------------------------------
# LLM Client abstraction
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt, user_prompt):
        """Send a chat completion request and return the assistant's text."""


class AzureOpenAIClient(LLMClient):
    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = os.environ["AZURE_OPENAI_BASE"]
        openai.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]
        self.engine = os.environ.get("AZURE_OPENAI_ENGINE", "gpt-4o")
        logger.info("AzureOpenAIClient initialized (engine=%s)", self.engine)

    def complete(self, system_prompt, user_prompt):
        response = openai.ChatCompletion.create(
            engine=self.engine,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4166,
            n=1,
            stop=None,
            temperature=1,
        )
        return response["choices"][0]["message"]["content"].strip()


class OllamaClient(LLMClient):
    def __init__(self):
        from langchain_ollama import ChatOllama

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3.3:latest")
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=1)
        logger.info("OllamaClient initialized (model=%s, base_url=%s)", model, base_url)

    def complete(self, system_prompt, user_prompt):
        response = self.llm.invoke(
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.content.strip()


class QwenClient(LLMClient):
    def __init__(self):
        openai.api_key = os.environ["DASHSCOPE_API_KEY"]
        openai.api_base = os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # Clear azure-specific settings if they were set
        openai.api_type = "open_ai"
        openai.api_version = ""
        self.model = os.environ.get("DASHSCOPE_MODEL", "qwen-max")
        logger.info("QwenClient initialized (model=%s)", self.model)

    def complete(self, system_prompt, user_prompt):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()


def create_llm_client(name):
    """Factory: create an LLM client by name."""
    clients = {
        "gpt4": AzureOpenAIClient,
        "llama": OllamaClient,
        "qwen": QwenClient,
    }
    if name not in clients:
        raise ValueError(f"Unknown LLM backend: {name}. Choose from {list(clients)}")
    return clients[name]()


# ---------------------------------------------------------------------------
# CQ generation
# ---------------------------------------------------------------------------

def generate_questions(client, rows):
    """Generate competency questions for a list of triple rows.

    Args:
        client: An LLMClient instance.
        rows: List of [subject, predicate, object] lists.

    Returns:
        List of generated question strings.
    """
    questions = []
    for i, row in enumerate(rows):
        user_prompt = f"{','.join(str(v) for v in row)}?"
        try:
            question = client.complete(SYSTEM_PROMPT, user_prompt)
            logger.info("Row %d/%d: generated CQs", i + 1, len(rows))
            questions.append(question)
        except Exception as e:
            logger.error("Row %d/%d error: %s", i + 1, len(rows), e)
            questions.append("Error generating question")
    return questions


def generate_questions_from_csv(client, triples_csv, output_xlsx):
    """Read triples CSV, generate CQs, write to Excel.

    Args:
        client: An LLMClient instance.
        triples_csv: Path to tab-delimited triples CSV.
        output_xlsx: Path to output Excel file.

    Returns:
        The output Excel path.
    """
    logger.info("Reading triples from %s", triples_csv)
    df = pd.read_csv(triples_csv, sep="\t", header=None, names=["Subject", "Predicate", "Object"])
    logger.info("Loaded %d triples", len(df))

    rows = df.values.tolist()
    questions = generate_questions(client, rows)
    df["Question"] = questions

    os.makedirs(os.path.dirname(output_xlsx) or ".", exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name="Questions", index=False)

    logger.info("Generated CQs written to %s", output_xlsx)
    return output_xlsx
