import json
import os
import re

TARGET_RELATIONS = ["instance of", "subclass of"]

ROOT = os.path.join(os.path.dirname(__file__), "..")
REBEL_RE = re.compile(r"<sub>\s*(.*?)\s*<rel>\s*(.*?)\s*<obj>\s*(.*?)\s*<et>")

files = [
    "data_rebel/rebel/en_train.jsonl",
    "data_rebel/rebel/en_val.jsonl",
    "data_rebel/rebel/en_test.jsonl",
]

counts = {rel: {"sentences": 0, "only_one": 0} for rel in TARGET_RELATIONS}

for filepath in files:
    with open(os.path.join(ROOT, filepath), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            answer = record.get("output", [{}])[0].get("answer")
            if not answer:
                continue
            triples = REBEL_RE.findall(answer)
            predicates = [t[1].lower() for t in triples]
            for rel in TARGET_RELATIONS:
                if rel in predicates:
                    counts[rel]["sentences"] += 1
                    if len(triples) == 1:
                        counts[rel]["only_one"] += 1

print(f"{'Relation':<16}  sentences  only_one")
for rel in TARGET_RELATIONS:
    print(f"{rel:<16}  {counts[rel]['sentences']:>9,}  {counts[rel]['only_one']:>8,}")
