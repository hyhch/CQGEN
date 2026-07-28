import logging

import pandas as pd

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = '''Simple CQs (Simple CQs)
features:
Single hop query: Only one ontology class or attribute needs to be accessed, without the need for cross relationship inference.
Direct retrieval: The answer can be obtained by directly matching attributes or instances in the ontology.

No computation or aggregation: does not involve operations such as statistics, sorting, conditional filtering, etc.

Example:
What is the username of the player?
(Directly retrieve the player's username attribute)

What is the genre of the game?
(Directly retrieve the type attributes of the game)

Which devices measure temperature?
(Directly match sensor classes and their measurement attributes)

What is a building?
(Directly retrieve the class definition of the ontology)

Complex CQs (Complex Problems)
features:
Multi hop query: requires crossing multiple ontology classes, attributes, or relationships (topological distance >= 2).
Inference or computation: involving conditional filtering, statistics, aggregation (such as "maximum", "average", "probability"), path analysis, etc.

Dynamic or cross domain: may rely on time series, spatial relationships, or cross ontology associations.

Example:
Who are the friends that play other games with this player?
(Need to associate the "player -> friend -> game" multi hop relationship)

What is the likelihood that a player who purchased in-app items in one game will do so in another?
(Probability Reasoning and Cross Game Behavior Analysis)

Which roads connect two towns via the optimum path?
(Spatial Path Planning and Multi condition Filtering)

How many players clicked an in-game advertisement and then started another game?
(Behavior Sequence Analysis and Statistics)

What are the most traded items in the game's marketplace?
(Aggregation statistics and sorting)
 Classify the following competency question as "Simple" or "Complex": {question}
Please do not reply any other information.'''


def classify_question(client, question):
    """Classify a single CQ as Simple or Complex using an LLM.

    Args:
        client: An LLMClient instance (from generate_cqs.py).
        question: The competency question string.

    Returns:
        "Simple", "Complex", or "Unknown".
    """
    try:
        response = client.complete(
            "You are a helpful assistant.",
            CLASSIFICATION_PROMPT.format(question=question),
        )
        response_lower = response.lower()
        if "simple" in response_lower:
            return "Simple"
        elif "complex" in response_lower:
            return "Complex"
        else:
            return "Unknown"
    except Exception as e:
        logger.error("Error classifying question '%s': %s", question[:50], e)
        return "Error"


def label_cqs_file(client, input_xlsx, output_xlsx, column="Sentence1"):
    """Label all CQs in an Excel file as Simple or Complex.

    Args:
        client: An LLMClient instance.
        input_xlsx: Path to input Excel file.
        output_xlsx: Path to output labeled Excel file.
        column: Column name containing the CQs to classify.

    Returns:
        The output file path.
    """
    logger.info("Labeling CQs from %s (column=%s)", input_xlsx, column)
    df = pd.read_excel(input_xlsx)

    if column not in df.columns:
        # Try alternative column names
        for alt in ["Competency Questions", "Sentence1", "Question"]:
            if alt in df.columns:
                column = alt
                break
        else:
            raise ValueError(
                f"Column '{column}' not found. Available: {df.columns.tolist()}"
            )

    labels = []
    for i, q in enumerate(df[column]):
        q_str = str(q).strip()
        if not q_str or q_str == "nan":
            labels.append("Unknown")
            continue
        label = classify_question(client, q_str)
        logger.info("  %d/%d: %s -> %s", i + 1, len(df), q_str[:60], label)
        labels.append(label)

    df["label"] = labels
    df.to_excel(output_xlsx, index=False)
    logger.info("Labeled CQs saved to %s", output_xlsx)
    return output_xlsx
