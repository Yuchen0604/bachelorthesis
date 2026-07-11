import argparse
import csv
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
tum_dir  = os.path.join(base_dir, "..", "data_tum", "tum")
out_dir  = os.path.join(base_dir, "relations_tum")
os.makedirs(out_dir, exist_ok=True)

VERSIONS = {
    "p99_new": {
        "csv":     os.path.join(out_dir, "re_data_relations_new.csv"),
        "dataset": os.path.join(tum_dir, "wiki_s2t_p99_new.jsonl"),
        "output":  os.path.join(out_dir, "tum_relations_wikidata_p99_new.json"),
    },
    "full": {
        "csv":     os.path.join(out_dir, "re_data_relations.csv"),
        "dataset": os.path.join(tum_dir, "wiki_s2t_full.jsonl"),
        "output":  os.path.join(out_dir, "tum_relations_wikidata.json"),
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=list(VERSIONS.keys()), required=True)
args = parser.parse_args()

cfg = VERSIONS[args.version]

# Collect relations actually present in the dataset
dataset_relations = set()
with open(cfg["dataset"], encoding="utf-8") as f:
    for line in f:
        for t in json.loads(line)["triples"]:
            dataset_relations.add(t["predicate_label"])
print(f"Relations in dataset: {len(dataset_relations)}")

# Convert CSV → JSON, keeping only relations present in the dataset
result = []
skipped = []
with open(cfg["csv"], encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["predicate_label"].strip()
        if label not in dataset_relations:
            skipped.append(label)
            continue
        result.append({
            "predicate_label":      label,
            "predicate_id":         row["predicate_id"],
            "wikidata_description": row["predicate_description"],
        })

with open(cfg["output"], "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved {len(result)} relations → {cfg['output']}")
print(f"Skipped {len(skipped)} (not in dataset): {skipped}")
missing_desc = [r for r in result if not r["wikidata_description"]]
missing_id   = [r for r in result if not r["predicate_id"]]
print(f"Missing description: {len(missing_desc)}")
print(f"Missing predicate ID: {len(missing_id)}")
