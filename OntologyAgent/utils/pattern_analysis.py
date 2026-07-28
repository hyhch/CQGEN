import json
from sentence_transformers import SentenceTransformer, util

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_cq_pred(cq_pred_list: list) -> list:
    # if competency question starts with "[Simple] / [Intermediate] / [Complex]", remove it
    output_cq_list = list()
    for question in cq_pred_list:
        if any(prefix in question for prefix in ["[Simple]", "[Intermediate]", "[Complex]"]):
            question = question.split("] ", 1)[-1].strip()
        output_cq_list.append(question)
    return output_cq_list

if __name__ == "__main__":
    ontology_name = "demcare"
    result_file_name = "results_gpt4o.json"

    # read pattern
    with open(f"./dataset/{ontology_name}/{ontology_name}_label.json", "r", encoding="utf-8") as f:
        pattern_data = json.load(f)
    # read generated competency questions
    with open(f"./dataset/{ontology_name}/{result_file_name}", "r", encoding="utf-8") as f:
    # with open("/home/lifeng/codes/OntologyAgent/llm4ke_qwen.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cq_pred_list = preprocess_cq_pred(data["retrofitted_competency_questions"])
    pred_embeddings = EMBEDDING_MODEL.encode(cq_pred_list, convert_to_tensor=True)
    
    output_dict = {
        "cqs": [],
        "label": [],
        "output": []
    }
    # 统计有多少种pattern，每种的CQ数量
    pattern_count = dict()
    pattern_match_count = dict()
    for item in pattern_data:
        pattern = item["label"]
        if pattern not in pattern_count:
            pattern_count[pattern] = 0
            pattern_match_count[pattern] = 0
        pattern_count[pattern] += 1

        gt_embeddings = EMBEDDING_MODEL.encode(item["question"], convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(pred_embeddings, gt_embeddings)
        output_dict["cqs"].append(item["question"])
        output_dict["label"].append(pattern)
        if max(cosine_scores) > 0.6:
            pattern_match_count[pattern] += 1
            output_dict["output"].append("matched")
        else:
            output_dict["output"].append("not matched")
    for key, value in pattern_count.items():
        print(f"{key}: {pattern_match_count[key]} / {value} = {pattern_match_count[key] / value:.3f}")

    # write to xlsx
    import pandas as pd
    df = pd.DataFrame(output_dict)
    df.to_excel(f"./{ontology_name}_label_analysis.xlsx", index=False)