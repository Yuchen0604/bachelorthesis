import json
import os
import re
from collections import Counter

base_dir    = os.path.dirname(os.path.abspath(__file__))
lagrange_dir = os.path.join(base_dir, "..", "data_lagrange", "lagrange")
output      = os.path.join(base_dir, "relations_lagrange", "lagrange_relations.json")

TRIPLE_RE = re.compile(r"<S>(.*?)<P>(.*?)<O>.*?(?=<sep>|$)")
FILES = ["lagrange_train.json", "lagrange_test.json"]


def parse_relations(triple_str: str) -> list[str]:
    return [m.group(2).strip() for m in TRIPLE_RE.finditer(triple_str) if m.group(2).strip()]


def main():
    counts: Counter = Counter()

    for filename in FILES:
        path = os.path.join(lagrange_dir, filename)
        print(f"Scanning {filename} ...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                triple_str = record.get("triples", "")
                if triple_str:
                    for rel in parse_relations(triple_str):
                        counts[rel] += 1
        print(f"  {len(counts):,} unique relations so far")

    result = [
        {"predicate_label": label, "predicate_id": "", "count": count}
        for label, count in counts.most_common()
    ]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(result):,} relations → {output}")


if __name__ == "__main__":
    main()
