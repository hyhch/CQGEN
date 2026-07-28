## Dataset
DATA_DIR_PATH = "./dataset"
ONTOLOGY_NAME = "vicinitycore"

## LLM Setting (DashScope / qwen-max as default, using OpenAI-compatible API)
LLM_API_TYPE = "openai"
LLM_API_KEY = "YOUR_API_KEY_HERE"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-max"

## Pipeline Parameters
CQ_EXAMPLES_NUM = 5
MAX_LOOP_COUNT = 3
MAX_EVAL_ITER = 3
MAX_NUM_SUB_TRIPLES = 25  # max triples per subgraph (controls segmentation granularity)

## Experiment Parameters
# Segmentation algorithm: "auto" (LLM selects) | "metis" | "louvain" | "leiden" | "spectral" | "random"
SEGMENTATION_METHOD = "auto"
# Random seed for reproducibility (affects "random" segmentation method and CQ example sampling)
RANDOM_SEED = 42
