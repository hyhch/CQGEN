"""Unified LLM backend: all models accessed via OpenAI-compatible API through LangChain."""

import logging
import os
import re

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)


def load_llm_config(config_path='config.yml'):
    """Load LLM backend configurations from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_llm(llm_name, config_path='config.yml'):
    """Create a LangChain ChatOpenAI instance for the named LLM backend.

    The config file should define each backend with:
      - base_url: API endpoint
      - model: model name
      - api_key_env: environment variable name holding the API key
    """
    config = load_llm_config(config_path)
    llm_cfg = config['llms'][llm_name]

    api_key = os.environ.get(llm_cfg.get('api_key_env', ''), '')
    if not api_key:
        api_key = llm_cfg.get('api_key', 'no-key')
        if api_key != 'no-key':
            log.warning("Using api_key from config file. Prefer setting %s env var.",
                        llm_cfg.get('api_key_env', ''))

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=llm_cfg['base_url'],
        model_name=llm_cfg['model'],
    )


def generate_cqs(llm, prompt_template, input_batches):
    """Send batches to the LLM and return raw text responses."""
    output_parser = StrOutputParser()
    prompt = PromptTemplate(
        input_variables=list(prompt_template.input),
        template=prompt_template.get(),
    )
    chain = prompt | llm | output_parser

    log.info("Sending %d batches to LLM", len(input_batches))
    return chain.batch(input_batches)


def parse_cqs(raw_responses):
    """Extract individual CQ strings from LLM text responses.

    Handles numbered lists (1. ...), bullet lists (- ...), and CQ-prefixed lines.
    """
    patterns = [
        r'(\d+)\.\s+(.+?)\?',    # "1. What is ...?"
        r'\s*-\s+(.+?)\?',       # "- What is ...?"
        r'CQ\d+:\s+(.+?)\n',     # "CQ1: What is ..."
    ]
    cqs = []
    for response in raw_responses:
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                for match in matches:
                    # Numbered pattern returns (number, question) tuple
                    cq = match[1] if isinstance(match, tuple) else match
                    cqs.append(cq.strip() + '?')
    return cqs
