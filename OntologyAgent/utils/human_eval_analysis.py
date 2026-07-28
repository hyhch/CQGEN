"""
Human Evaluation Analysis for CQGen-MAS

Reads two annotated sheets + answer key, computes:
  1. Inter-annotator agreement (Cohen's kappa) per dimension
  2. Per-dataset Valid% for unmatched (FP) and matched (TP) CQs
  3. Overall statistics
  4. Adjusted Precision

Usage:
    python utils/human_eval_analysis.py
"""

import csv
import os
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "human_eval")


def read_annotations(path):
    """Read annotated CSV, return dict: ID -> (fluency, relevance, answerability) as 1/0."""
    annotations = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cq_id = int(row["ID"])
            flu = 1 if row["Fluency (Y/N)"].strip().upper() == "Y" else 0
            rel = 1 if row["Relevance (Y/N)"].strip().upper() == "Y" else 0
            ans = 1 if row["Answerability (Y/N)"].strip().upper() == "Y" else 0
            annotations[cq_id] = (flu, rel, ans)
    return annotations


def read_answer_key(path):
    """Read answer key CSV, return dict: ID -> {dataset, match_type, llm}."""
    key = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cq_id = int(row["ID"])
            key[cq_id] = {
                "dataset": row["Dataset"],
                "match_type": row["Match Type"],
                "llm": row["Source LLM"],
            }
    return key


def cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two binary label lists."""
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return float("nan")

    # Observed agreement
    agree = sum(a == b for a, b in zip(y1, y2))
    po = agree / n

    # Expected agreement
    p1_pos = sum(y1) / n
    p2_pos = sum(y2) / n
    pe = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    # Load data
    ann1 = read_annotations(os.path.join(EVAL_DIR, "annotated_sheet1.csv"))
    ann2 = read_annotations(os.path.join(EVAL_DIR, "annotated_sheet2.csv"))
    key = read_answer_key(os.path.join(EVAL_DIR, "answer_key.csv"))

    ids = sorted(set(ann1.keys()) & set(ann2.keys()) & set(key.keys()))
    print(f"Total CQs with annotations from both annotators: {len(ids)}")

    # =========================================================================
    # 1. Inter-Annotator Agreement (Cohen's kappa)
    # =========================================================================
    dims = ["Fluency", "Relevance", "Answerability"]
    for d_idx, dim in enumerate(dims):
        y1 = [ann1[i][d_idx] for i in ids]
        y2 = [ann2[i][d_idx] for i in ids]
        agree_pct = sum(a == b for a, b in zip(y1, y2)) / len(ids) * 100
        kappa = cohens_kappa(y1, y2)
        print(f"  {dim:15s}  Agreement: {agree_pct:5.1f}%  Cohen's κ: {kappa:.3f}")

    # Compute "Valid" (all three Y) agreement
    valid1 = [1 if all(ann1[i]) else 0 for i in ids]
    valid2 = [1 if all(ann2[i]) else 0 for i in ids]
    agree_valid = sum(a == b for a, b in zip(valid1, valid2)) / len(ids) * 100
    kappa_valid = cohens_kappa(valid1, valid2)
    print(f"  {'Valid (all Y)':15s}  Agreement: {agree_valid:5.1f}%  Cohen's κ: {kappa_valid:.3f}")

    # =========================================================================
    # 2. Majority vote: CQ is Y if both annotators say Y (strict)
    #    We use "both agree Y" as the criterion for each dimension
    # =========================================================================
    print(f"\n{'='*80}")
    print("Per-dataset results (majority vote: both annotators agree Y)")
    print(f"{'='*80}")

    # Organize by dataset and match_type
    datasets = ["OneM2M", "SAREF4ENV", "VGO", "VC"]
    results = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "fluency": 0, "relevance": 0, "answerability": 0, "valid": 0
    }))

    for i in ids:
        ds = key[i]["dataset"]
        mt = key[i]["match_type"]

        # Majority vote (both agree Y)
        flu = 1 if ann1[i][0] == 1 and ann2[i][0] == 1 else 0
        rel = 1 if ann1[i][1] == 1 and ann2[i][1] == 1 else 0
        ans = 1 if ann1[i][2] == 1 and ann2[i][2] == 1 else 0
        valid = 1 if flu and rel and ans else 0

        results[ds][mt]["total"] += 1
        results[ds][mt]["fluency"] += flu
        results[ds][mt]["relevance"] += rel
        results[ds][mt]["answerability"] += ans
        results[ds][mt]["valid"] += valid

    # Print table
    print(f"\n{'Dataset':<12} {'Group':<12} {'N':>4} {'Fluency%':>10} {'Relevance%':>12} {'Answer.%':>10} {'Valid%':>8}")
    print("-" * 72)

    overall_unmatched = {"total": 0, "fluency": 0, "relevance": 0, "answerability": 0, "valid": 0}
    overall_matched = {"total": 0, "fluency": 0, "relevance": 0, "answerability": 0, "valid": 0}

    for ds in datasets:
        for mt in ["matched", "unmatched"]:
            r = results[ds][mt]
            n = r["total"]
            if n == 0:
                continue
            pf = r["fluency"] / n * 100
            pr = r["relevance"] / n * 100
            pa = r["answerability"] / n * 100
            pv = r["valid"] / n * 100
            print(f"{ds:<12} {mt:<12} {n:>4} {pf:>9.1f}% {pr:>11.1f}% {pa:>9.1f}% {pv:>7.1f}%")

            target = overall_unmatched if mt == "unmatched" else overall_matched
            for k in target:
                target[k] += r[k]

    print("-" * 72)
    for label, ov in [("matched", overall_matched), ("unmatched", overall_unmatched)]:
        n = ov["total"]
        if n == 0:
            continue
        print(f"{'Overall':<12} {label:<12} {n:>4} "
              f"{ov['fluency']/n*100:>9.1f}% "
              f"{ov['relevance']/n*100:>11.1f}% "
              f"{ov['answerability']/n*100:>9.1f}% "
              f"{ov['valid']/n*100:>7.1f}%")

    # =========================================================================
    # 3. Key finding: How many "false positives" are actually valid?
    # =========================================================================
    print(f"\n{'='*80}")
    print("KEY FINDING")
    print(f"{'='*80}")
    n_um = overall_unmatched["total"]
    n_valid_um = overall_unmatched["valid"]
    pct = n_valid_um / n_um * 100 if n_um > 0 else 0
    print(f"Of {n_um} unmatched CQs (judged as 'false positives' by automated evaluation):")
    print(f"  {n_valid_um} ({pct:.1f}%) are judged VALID by human annotators")
    print(f"  {n_um - n_valid_um} ({100-pct:.1f}%) are genuinely invalid")

    # =========================================================================
    # 4. Adjusted Precision (per dataset)
    # =========================================================================
    print(f"\n{'='*80}")
    print("Adjusted Precision Estimate")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<12} {'Orig P':>8} {'FP_valid%':>10} {'P_adj (est)':>12}")
    print("-" * 45)

    # Original precision from paper (averaged over 3 LLMs)
    orig_precision = {
        "OneM2M": 0.53, "SAREF4ENV": 0.30, "VGO": 0.56, "VC": 0.22
    }

    for ds in datasets:
        r_um = results[ds]["unmatched"]
        n_um = r_um["total"]
        n_valid = r_um["valid"]
        valid_rate = n_valid / n_um if n_um > 0 else 0

        p_orig = orig_precision.get(ds, 0)

        # Adjusted precision:
        # Original: P = TP / (TP + FP)
        # TP = P * total_generated, FP = (1-P) * total_generated
        # FP_valid = valid_rate * FP
        # P_adj = (TP + FP_valid) / (TP + FP_valid + FP_invalid)
        #       = (TP + valid_rate * FP) / (TP + valid_rate * FP + (1-valid_rate) * FP)
        #       = (P + valid_rate * (1-P)) / (P + valid_rate*(1-P) + (1-valid_rate)*(1-P))
        #       = (P + valid_rate * (1-P)) / 1
        #       = P + valid_rate * (1 - P)
        p_adj = p_orig + valid_rate * (1 - p_orig)

        print(f"{ds:<12} {p_orig:>8.2f} {valid_rate*100:>9.1f}% {p_adj:>11.2f}")

    # Overall (use per-dataset valid rates to compute average adjusted precision)
    per_ds_adj = []
    per_ds_vr = []
    for ds in datasets:
        r_um = results[ds]["unmatched"]
        n = r_um["total"]
        vr = r_um["valid"] / n if n > 0 else 0
        p = orig_precision.get(ds, 0)
        per_ds_adj.append(p + vr * (1 - p))
        per_ds_vr.append(vr)
    avg_p = sum(orig_precision.values()) / len(orig_precision)
    avg_vr = sum(per_ds_vr) / len(per_ds_vr)
    avg_p_adj = sum(per_ds_adj) / len(per_ds_adj)
    print("-" * 45)
    print(f"{'Average':<12} {avg_p:>8.2f} {avg_vr*100:>9.1f}% {avg_p_adj:>11.2f}")

    # =========================================================================
    # 5. Disagreement examples (for qualitative analysis)
    # =========================================================================
    print(f"\n{'='*80}")
    print("Disagreement examples (annotators differ on Valid)")
    print(f"{'='*80}")
    disagree_count = 0
    for i in ids:
        v1 = 1 if all(ann1[i]) else 0
        v2 = 1 if all(ann2[i]) else 0
        if v1 != v2:
            disagree_count += 1
            if disagree_count <= 5:
                ds = key[i]["dataset"]
                mt = key[i]["match_type"]
                # Find the CQ text from one of the annotated sheets
                print(f"\n  ID={i} [{ds}, {mt}]")
                print(f"    Ann1: F={ann1[i][0]} R={ann1[i][1]} A={ann1[i][2]}")
                print(f"    Ann2: F={ann2[i][0]} R={ann2[i][1]} A={ann2[i][2]}")
    print(f"\n  Total disagreements on Valid: {disagree_count}/{len(ids)}")


if __name__ == "__main__":
    main()
