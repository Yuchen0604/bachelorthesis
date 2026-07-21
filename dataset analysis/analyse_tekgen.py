import json
import os

TARGET_RELATIONS = ["instance of", "subclass of"]

ROOT = os.path.join(os.path.dirname(__file__), "..")

files = [
    "data_tekgen/tekgen/quadruples-train.tsv",
    "data_tekgen/tekgen/quadruples-validation.tsv",
    "data_tekgen/tekgen/quadruples-test.tsv",
]

counts = {rel: {"sentences": 0, "only_one": 0} for rel in TARGET_RELATIONS}

for filepath in files:
    with open(os.path.join(ROOT, filepath), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triples = record.get("triples", [])
            triples = [t for t in triples if len(t) == 3]
            predicates = [t[1].lower() for t in triples]
            for rel in TARGET_RELATIONS:
                if rel in predicates:
                    counts[rel]["sentences"] += 1
                    if len(triples) == 1:
                        counts[rel]["only_one"] += 1

print(f"{'Relation':<16}  sentences  only_one")
for rel in TARGET_RELATIONS:
    print(f"{rel:<16}  {counts[rel]['sentences']:>9,}  {counts[rel]['only_one']:>8,}")
