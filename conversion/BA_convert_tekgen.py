import json
import os

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "data_tekgen", "re_data")
TEKGEN_DIR = os.path.join(ROOT_DIR, "data_tekgen", "tekgen")

SPLITS = {
    "train": "quadruples-train.tsv",
    "valid": "quadruples-validation.tsv",
    "test":  "quadruples-test.tsv",
}


def convert(input_path: str, output_path: str) -> int:
    written = skipped_no_triples = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            #keep only 3-element triples and skip 4-element quadruples
            triples_raw = [t for t in record.get("triples", []) if len(t) == 3]
            if not triples_raw:
                skipped_no_triples += 1
                continue

            triples = [
                {
                    "triple_index":    i,
                    "subject_id":      "",
                    "subject_label":   subj,
                    "predicate_id":    "",
                    "predicate_label": rel,
                    "object_id":       "",
                    "object_label":    obj,
                }
                for i, (subj, rel, obj) in enumerate(triples_raw)
            ]

            fout.write(json.dumps({
                "category": "tekgen",
                "entity":   "",
                "sentence": record.get("sentence", ""),
                "triples":  triples,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {written:>8,} written  →  {output_path}")
    print(f"    skipped {skipped_no_triples:>6,}  — no valid triples (all quadruples or empty)")
    return written


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split, filename in SPLITS.items():
        input_path  = os.path.join(TEKGEN_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"tekgen_{split}.jsonl")
        print(f"Converting {split} ...")
        convert(input_path, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
