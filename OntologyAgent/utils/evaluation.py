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

def evaluate(cq_pred_list: list, cq_gt_list: list, threshold: float = 0.6) -> tuple:
    # Preprocess competency questions
    cq_pred_list = preprocess_cq_pred(cq_pred_list)

    # Compute embeddings
    pred_embeddings = EMBEDDING_MODEL.encode(cq_pred_list, convert_to_tensor=True)
    gt_embeddings = EMBEDDING_MODEL.encode(cq_gt_list, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(pred_embeddings, gt_embeddings)

    tp_pred_set = set()
    tp_gt_set = set()
    for i in range(len(cosine_scores)):
        for j in range(len(cosine_scores[0])):
            if cosine_scores[i][j] > threshold:
                tp_pred_set.add(i)
                tp_gt_set.add(j)

    print(f"{len(tp_pred_set)} / {len(cq_pred_list)}, {len(tp_gt_set)} / {len(cq_gt_list)}")
    precision = len(tp_pred_set) / len(cq_pred_list)
    recall = len(tp_gt_set) / len(cq_gt_list)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

if __name__ == "__main__":
    # Get generated competency questions
    input_file = "/home/lifeng/codes/OntologyAgent/dataset/onem2m/results.json"
    with open(input_file, 'r') as f:
        data = json.load(f)
    precision, recall, f1 = evaluate(data["retrofitted_competency_questions"], data["competency_questions"])
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}", f"F1: {f1:.4f}")
