import json
import re
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


# Normalisation
def normalize(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_triple(s, r, o):
    return (normalize(s), normalize(r), normalize(o))


#Computes token-level F1
def token_overlap_f1(pre_str, gol_str):
    if not pre_str and not gol_str:
        return 1.0
    if not pre_str or not gol_str:
        return 0.0
    tokens1 = pre_str.split()
    tokens2 = gol_str.split()
    overlap  = sum((Counter(tokens1) & Counter(tokens2)).values())
    precision = overlap / len(tokens1)
    recall    = overlap / len(tokens2)
    if overlap == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



#Four metric scores from slot overlaps
def metric_scores(sub_ov, pred_ov, obj_ov, switched, switch_type,
                   orig_sub_ov, orig_pred_ov, orig_obj_ov):
    # strict: per slot — exact boundary match AND correct role assignment (switched slots = type mismatch → 0)
    if not switched:
        strict = ((1.0 if sub_ov  == 1.0 else 0.0) +
                  (1.0 if pred_ov == 1.0 else 0.0) +
                  (1.0 if obj_ov  == 1.0 else 0.0)) / 3.0
    elif switch_type == "sub_obj":
        strict = (1.0 if pred_ov == 1.0 else 0.0) / 3.0
    elif switch_type == "sub_pred":
        strict = (1.0 if obj_ov  == 1.0 else 0.0) / 3.0
    else:  # "pred_obj"
        strict = (1.0 if sub_ov  == 1.0 else 0.0) / 3.0

    ent_type = ((1.0 if orig_sub_ov  > 0 else 0.0) +
                (1.0 if orig_pred_ov > 0 else 0.0) +
                (1.0 if orig_obj_ov  > 0 else 0.0)) / 3.0

    # exact: per slot — exact boundary match regardless of role (switching does not penalise)
    exact = ((1.0 if sub_ov  == 1.0 else 0.0) +
             (1.0 if pred_ov == 1.0 else 0.0) +
             (1.0 if obj_ov  == 1.0 else 0.0)) / 3.0

    partial  = (sub_ov + pred_ov + obj_ov) / 3.0
    return {
        "strict":   round(strict,   4),
        "ent_type": round(ent_type, 4),
        "exact":    round(exact,    4),
        "partial":  round(partial,  4),
    }



# Step 1: compare one predicted triple against one gold triple

def compare_triple_pair(pred, gold):
    ps, pr, po = pred   #predicted subject, predicate, object
    gs, gr, go = gold   # gold subject, predicate, object

    # compute slot overlaps in the original positions
    orig_sub_ov  = token_overlap_f1(ps, gs)
    orig_pred_ov = token_overlap_f1(pr, gr)
    orig_obj_ov  = token_overlap_f1(po, go)

    sub_ov, pred_ov, obj_ov = orig_sub_ov, orig_pred_ov, orig_obj_ov

    sub_found  = orig_sub_ov  > 0
    pred_found = orig_pred_ov > 0
    obj_found  = orig_obj_ov  > 0

    switched            = False
    switch_type         = None
    scores_before_switch = None


    #Swap 1: subject <-> object
    if not sub_found and not obj_found:
        new_sub_ov = token_overlap_f1(ps, go)
        new_obj_ov = token_overlap_f1(po, gs)
        if new_sub_ov > 0 or new_obj_ov > 0:
            scores_before_switch = metric_scores(
                orig_sub_ov, orig_pred_ov, orig_obj_ov, False, None,
                orig_sub_ov, orig_pred_ov, orig_obj_ov)
            sub_ov, obj_ov = new_sub_ov, new_obj_ov
            switched, switch_type = True, "sub_obj"

    #Swap 2: subject <-> predicate
    if not switched and not sub_found and not pred_found:
        new_sub_ov  = token_overlap_f1(ps, gr)
        new_pred_ov = token_overlap_f1(pr, gs)
        if new_sub_ov > 0 or new_pred_ov > 0:
            scores_before_switch = metric_scores(
                orig_sub_ov, orig_pred_ov, orig_obj_ov, False, None,
                orig_sub_ov, orig_pred_ov, orig_obj_ov)
            sub_ov, pred_ov = new_sub_ov, new_pred_ov
            switched, switch_type = True, "sub_pred"

    #swap 3: predicate <-> object
    if not switched and not pred_found and not obj_found:
        new_pred_ov = token_overlap_f1(pr, go)
        new_obj_ov  = token_overlap_f1(po, gr)
        if new_pred_ov > 0 or new_obj_ov > 0:
            scores_before_switch = metric_scores(
                orig_sub_ov, orig_pred_ov, orig_obj_ov, False, None,
                orig_sub_ov, orig_pred_ov, orig_obj_ov)
            pred_ov, obj_ov = new_pred_ov, new_obj_ov
            switched, switch_type = True, "pred_obj"

    scores = metric_scores(sub_ov, pred_ov, obj_ov, switched, switch_type,
                             orig_sub_ov, orig_pred_ov, orig_obj_ov)

    return {
        "slot_scores": {
            "sub_overlap":  round(sub_ov,  4),
            "pred_overlap": round(pred_ov, 4),
            "obj_overlap":  round(obj_ov,  4),
        },
        "switched":             switched,
        "switch_type":          switch_type,
        "scores_before_switch": scores_before_switch,
        "scores":               scores,
    }



#step 2: align predicted triples to gold triples for one sample

METRICS = ("strict", "ent_type", "exact", "partial")


def align_and_score(pred_triples, gold_triples):
    n_pred = len(pred_triples)
    n_gold = len(gold_triples)

    if n_pred == 0 and n_gold == 0:
        return {
            "alignment":         [],
            "unmatched_gold":    [],
            "unmatched_pred":    [],
            "per_sample_scores": {m: {"precision": 1.0, "recall": 1.0, "f1": 1.0} for m in METRICS},
        }

    if n_pred == 0 or n_gold == 0:
        return {
            "alignment":         [],
            "unmatched_gold":    [{"gold_idx": j, "triple": list(normalize_triple(*t))} for j, t in enumerate(gold_triples)],
            "unmatched_pred":    [{"pred_idx": i, "triple": list(normalize_triple(*t))} for i, t in enumerate(pred_triples)],
            "per_sample_scores": {m: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for m in METRICS},
        }

    norm_pred = [normalize_triple(*t) for t in pred_triples]
    norm_gold = [normalize_triple(*t) for t in gold_triples]

    #Build an N x M score matrix where each cell is the average of all four metric F1 values for that pair. 
    pair_results = [[compare_triple_pair(norm_pred[i], norm_gold[j])
                     for j in range(n_gold)]
                    for i in range(n_pred)]

    score_matrix = np.array([
        [sum(pair_results[i][j]["scores"][m] for m in METRICS) / len(METRICS)
         for j in range(n_gold)]
        for i in range(n_pred)
    ])

    #Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    matched_pred = set(row_ind)
    matched_gold = set(col_ind)

    alignment = []
    for pi, gi in zip(row_ind, col_ind):
        result = pair_results[pi][gi]
        alignment.append({
            "pred_idx": int(pi),
            "gold_idx": int(gi),
            "pred":     list(norm_pred[pi]),
            "gold":     list(norm_gold[gi]),
            **result,
        })

    unmatched_pred = [{"pred_idx": i, "triple": list(norm_pred[i])} for i in range(n_pred) if i not in matched_pred]
    unmatched_gold = [{"gold_idx": j, "triple": list(norm_gold[j])} for j in range(n_gold) if j not in matched_gold]

    per_sample_scores = {}
    for m in METRICS:
        score_sum = sum(a["scores"][m] for a in alignment)
        precision = score_sum / n_pred
        recall    = score_sum / n_gold
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_sample_scores[m] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    return {
        "alignment":         alignment,
        "unmatched_gold":    unmatched_gold,
        "unmatched_pred":    unmatched_pred,
        "per_sample_scores": per_sample_scores,
    }



#Step 3: dataset-level evaluation

def evaluate_dataset(samples, eval_output_path="eval_output.jsonl", eval_summary_path="eval_summary.json"):
    per_sample_results = []

    webnlg_sums   = {m: 0.0 for m in METRICS}
    webnlg_n_pred = 0
    webnlg_n_gold = 0

    for sample in samples:
        pred_triples = sample["pred_triples"]
        gold_triples = sample["gold_triples"]

        result = align_and_score(pred_triples, gold_triples)

        record = {
            "id":                    sample.get("id"),
            "sentence":              sample.get("sentence"),
            "gold_triples_raw":      gold_triples,
            "gold_triples_normalized": [list(normalize_triple(*t)) for t in gold_triples],
            "pred_triples_raw":      pred_triples,
            "pred_triples_normalized": [list(normalize_triple(*t)) for t in pred_triples],
            "alignment":             result["alignment"],
            "unmatched_gold":        result["unmatched_gold"],
            "unmatched_pred":        result["unmatched_pred"],
            "per_sample_scores":     result["per_sample_scores"],
        }
        per_sample_results.append(record)

        for pair in result["alignment"]:
            for m in METRICS:
                webnlg_sums[m] += pair["scores"][m]
        webnlg_n_pred += len(pred_triples)
        webnlg_n_gold += len(gold_triples)

    with open(eval_output_path, "w", encoding="utf-8") as f:
        for record in per_sample_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    #Macro
    n = len(per_sample_results)
    macro = {}
    for m in METRICS:
        macro[m] = {
            "precision": round(sum(r["per_sample_scores"][m]["precision"] for r in per_sample_results) / n, 4),
            "recall":    round(sum(r["per_sample_scores"][m]["recall"]    for r in per_sample_results) / n, 4),
            "f1":        round(sum(r["per_sample_scores"][m]["f1"]        for r in per_sample_results) / n, 4),
        }

    # WebNLG-style
    webnlg = {}
    for m in METRICS:
        precision = webnlg_sums[m] / webnlg_n_pred if webnlg_n_pred > 0 else 0.0
        recall    = webnlg_sums[m] / webnlg_n_gold if webnlg_n_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        webnlg[m] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    summary = {
        "n_samples": n,
        "macro":       macro,
        "webNLG_style": webnlg,
    }

    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL from run_inference.py")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.predictions))[0]
    dataset_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.predictions)))
    results_dir = os.path.join(dataset_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.predictions, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]

    samples = [
        {
            "id":           r["id"],
            "sentence":     r.get("sentence"),
            "pred_triples": r["model_predicted_triples"],
            "gold_triples": r["gold_triples"],
        }
        for r in raw
    ]

    eval_output_path  = os.path.join(results_dir, f"{stem}_eval.jsonl")
    eval_summary_path = os.path.join(results_dir, f"{stem}.json")

    summary = evaluate_dataset(samples, eval_output_path, eval_summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
