import json
import os
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")

datasets = {
    "tum_p99_new": "data_tum/dataset-instruct-large-p99_new/train.jsonl",
    "rebel":       "data_rebel/dataset-instruct-20k/train.jsonl",
    "lagrange":    "data_lagrange/dataset-instruct-20k/train.jsonl",
    "tekgen":      "data_tekgen/dataset-instruct-20k/train.jsonl",
    "wikinre":     "data_wikinre/dataset-instruct-20k/train.jsonl",
}


def get_sentence(record):
    for msg in record["messages"]:
        if msg["role"] == "user":
            content = msg["content"]
            start = content.find('"')
            end = content.rfind('"')
            if start != -1 and end != start:
                return content[start+1:end]
    return None


def get_triples(record):
    for msg in record["messages"]:
        if msg["role"] == "assistant":
            return frozenset(msg["content"].strip().splitlines())
    return frozenset()


def normalize(sentence):
    #wikinre sentences are stored pre-tokenized
    s = re.sub(r"\s+([.,;:!?)\]])", r"\1", sentence)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


#normalized sentence
sentence_index = defaultdict(dict)
raw_sentence = defaultdict(dict)

for dataset, rel_path in datasets.items():
    path = os.path.join(ROOT, rel_path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sentence = get_sentence(record)
            if sentence:
                key = normalize(sentence)
                sentence_index[key][dataset] = get_triples(record)
                raw_sentence[key][dataset] = sentence

# to find sentences appearing in more than one dataset
overlaps = {s: d for s, d in sentence_index.items() if len(d) > 1}

print(f"Total unique sentences across all datasets: {len(sentence_index):,}")
print(f"Sentences appearing in 2+ datasets:         {len(overlaps):,}")

OTHER_DATASETS = ["rebel", "lagrange", "tekgen", "wikinre"]

pair_counts = defaultdict(int)
differ_counts = defaultdict(int)

output = []
for sentence, datasets in overlaps.items():
    if "tum_p99_new" not in datasets:
        continue
    for other in OTHER_DATASETS:
        if other in datasets:
            pair_counts[other] += 1
            same = datasets["tum_p99_new"] == datasets[other]
            if not same:
                differ_counts[other] += 1
            output.append({
                "compared_with": other,
                "same_triples": same,
                "tum_p99_new_sentence": raw_sentence[sentence]["tum_p99_new"],
                f"{other}_sentence": raw_sentence[sentence][other],
                "tum_p99_new": sorted(datasets["tum_p99_new"]),
                other: sorted(datasets[other]),
            })

print("\nOverlap of tum_p99_new vs other datasets:")
print(f"{'Dataset':<12}  {'shared':>8}  {'different triples':>17}")
for other in OTHER_DATASETS:
    print(f"{other:<12}  {pair_counts[other]:>8,}  {differ_counts[other]:>17,}")


output_path = os.path.join(os.path.dirname(__file__), "sentence_overlaps.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

