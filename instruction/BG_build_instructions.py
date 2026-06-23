import argparse
import json
import os

CN = 20

DATASET_CONFIG = {
    "rebel": {
        "data_dir": "./data_rebel/re_data",
        "out_dir":  "./data_rebel/dataset-instruct",
        "splits": {
            "rebel_train": "train",
            "rebel_val":   "valid",
            "rebel_test":  "test",
        },
    },
    "lagrange": {
        "data_dir": "./data_lagrange/re_data",
        "out_dir":  "./data_lagrange/dataset-instruct",
        "splits": {
            "lagrange_train": "train",
            "lagrange_test":  "test",
        },
    },
    "tekgen": {
        "data_dir": "./data_tekgen/re_data",
        "out_dir":  "./data_tekgen/dataset-instruct",
        "splits": {
            "tekgen_train": "train",
            "tekgen_valid": "valid",
            "tekgen_test":  "test",
        },
    },
}

SYSTEM_PROMPT = (
    "You are an information extraction assistant. "
    "Extract all valid relation triples from the sentence "
    "based on the given candidate relations."
)


def build_user_prompt(sentence, candidate_relations):
    candidate_str = "\n".join(f"- {r}" for r in candidate_relations)
    return (
        f"Sentence: \"{sentence}\"\n\n"
        f"Candidate relations:\n{candidate_str}\n\n"
        f"Extract all valid triples from the sentence.\n"
        f"Each triple on a new line in the format: subject | relation | object"
    )


def build_assistant_output(triples, candidate_relations):
    relation_order = {r: i for i, r in enumerate(candidate_relations)}
    sorted_triples = sorted(
        triples,
        key=lambda t: relation_order.get(t["predicate_label"], 999)
    )
    lines = [
        f"{t['subject_label']} | {t['predicate_label']} | {t['object_label']}"
        for t in sorted_triples
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rebel", "lagrange", "tekgen"], default="rebel")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    data_dir, out_dir, splits = cfg["data_dir"], cfg["out_dir"], cfg["splits"]

    LOG_DIR = "./logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log_lines = ["=" * 60, "BUILD INSTRUCTION DATA REPORT",
                 f"dataset = {args.dataset}", f"CN = {CN}", "=" * 60, ""]

    for split, out_name in splits.items():
        input_path  = os.path.join(data_dir, f"{split}_cn{CN}.jsonl")
        output_path = os.path.join(out_dir, f"{out_name}.jsonl")

        total = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total += 1
                output = {
                    "id": data["id"],
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": build_user_prompt(
                            data["sentence"], data["candidate_relations"])},
                        {"role": "assistant", "content": build_assistant_output(
                            data["triples"], data["candidate_relations"])},
                    ]
                }
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")

        log_lines += [f"[{split}]", f"  Input  : {input_path}",
                      f"  Output : {output_path}", f"  Samples: {total}", ""]

    log_content = "\n".join(log_lines)
    print(log_content)
    with open(os.path.join(LOG_DIR, f"build_instruction_data_{args.dataset}.log"), "w", encoding="utf-8") as f:
        f.write(log_content)


if __name__ == "__main__":
    main()