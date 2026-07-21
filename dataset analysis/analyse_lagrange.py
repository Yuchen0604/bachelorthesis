import json
import os
import re

TARGET_RELATIONS = ["instance of", "subclass of"]

ROOT = os.path.join(os.path.dirname(__file__), "..")
LAGRANGE_RE = re.compile(r"<S>(.*?)<P>(.*?)<O>(.*?)(?:<sep>|$)")

files = [
    "data_lagrange/lagrange/lagrange_train.json",
    "data_lagrange/lagrange/lagrange_test.json",
]

counts = {rel: {"sentences": 0, "only_one": 0} for rel in TARGET_RELATIONS}

for filepath in files:
    with open(os.path.join(ROOT, filepath), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            triples = LAGRANGE_RE.findall(record.get("triples", ""))
            predicates = [t[1].lower() for t in triples]
            for rel in TARGET_RELATIONS:
                if rel in predicates:
                    counts[rel]["sentences"] += 1
                    if len(triples) == 1:
                        counts[rel]["only_one"] += 1

print(f"{'Relation':<16}  sentences  only_one")
for rel in TARGET_RELATIONS:
    print(f"{rel:<16}  {counts[rel]['sentences']:>9,}  {counts[rel]['only_one']:>8,}")
