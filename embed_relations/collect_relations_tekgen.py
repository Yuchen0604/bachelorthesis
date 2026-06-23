import json
import os
from collections import Counter

base_dir   = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(base_dir, "..", "data_tekgen")
output_dir = os.path.join(base_dir, "relations_tekgen")
output_all = os.path.join(output_dir, "tekgen_relations.json")

splits = ["train", "validation", "test"]


def main():
    counts: Counter = Counter()

    for split in splits:
        path = os.path.join(data_dir, f"quadruples-{split}.tsv")
        print(f"Scanning quadruples-{split}.tsv ...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for triple in record.get("triples", []):
                    if len(triple) != 3:
                        continue
                    predicate = triple[1]
                    if predicate:
                        counts[predicate] += 1
        print(f"  {len(counts):,} unique relations so far")

    result = [
        {"predicate_label": label, "predicate_id": "", "count": count}
        for label, count in counts.most_common()
    ]

    os.makedirs(output_dir, exist_ok=True)
    with open(output_all, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(result):,} relations to {output_all}")

if __name__ == "__main__":
    main()
