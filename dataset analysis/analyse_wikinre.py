import json
import os

TARGET_RELATIONS = ["instance of", "subclass of"]

ROOT = os.path.join(os.path.dirname(__file__), "..")

files = [
    "data_wikinre/wikinre/train.jsonl",
    "data_wikinre/wikinre/val.jsonl",
    "data_wikinre/wikinre/test.jsonl",
]

counts = {rel: {"sentences": 0, "only_one": 0} for rel in TARGET_RELATIONS}

for filepath in files:
    with open(os.path.join(ROOT, filepath), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triplets = record.get("triplets", [])
            predicates = [t["predicate"]["surfaceform"].lower() for t in triplets]
            for rel in TARGET_RELATIONS:
                if rel in predicates:
                    counts[rel]["sentences"] += 1
                    if len(triplets) == 1:
                        counts[rel]["only_one"] += 1

print(f"{'Relation':<16}  sentences  only_one")
for rel in TARGET_RELATIONS:
    print(f"{rel:<16}  {counts[rel]['sentences']:>9,}  {counts[rel]['only_one']:>8,}")
