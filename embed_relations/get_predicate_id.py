import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.parse

base_dir = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    "lagrange": {
        "input":   os.path.join(base_dir, "relations_lagrange", "lagrange_relations.json"),
        "found":   os.path.join(base_dir, "relations_lagrange", "lagrange_relations_predicate_id.json"),
        "missing": os.path.join(base_dir, "relations_lagrange", "lagrange_relations_missing.json"),
    },
    "tekgen": {
        "input":   os.path.join(base_dir, "relations_tekgen", "tekgen_relations.json"),
        "found":   os.path.join(base_dir, "relations_tekgen", "tekgen_relations_predicate_id.json"),
        "missing": os.path.join(base_dir, "relations_tekgen", "tekgen_relations_missing.json"),
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="lagrange")
args = parser.parse_args()

cfg = DATASET_CONFIG[args.dataset]
INPUT_FILE          = cfg["input"]
OUTPUT_FILE         = cfg["found"]
OUTPUT_FILE_MISSING = cfg["missing"]

all_rebel_relations = os.path.join(base_dir, "relations_rebel", "nonorig_relations.json")
relations_yuchen    = os.path.join(base_dir, "..", "predicate_descriptions.csv")
entities_jsonl      = os.path.join(base_dir, "..", "entities.jsonl")

wikidata_api = "https://www.wikidata.org/w/api.php"
API_DELAY    = 0.3


manual_map = {
    "number of speakers": "P1098",
    "work period (end)": "P2032",
    "students count": "P2196",
    "instance has part(s) of the class": "P2670",
    "crew member(s)": "P1029",
    "Roman nomen gentilicium": "P2359",
    "number of seats in legislature": "P1410",
    "professional name (Japan)": "P2838",
    "animal species kept": "P1990",
    "statistical unit": "P2353",
    "vertex figure": "P1678",
    "World Health Organisation international non-proprietary name": "P2275",
    "periapsis date": "P11796",
    "term length of office" : "P2097",
    "national flower" : "P2238",
    "fracturing": "P538"
}


def load_nonorig_map(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for e in data:
        label = e.get("predicate_label") or e.get("label")
        pid   = e.get("predicate_id")   or e.get("id")
        if label and pid:
            result[label] = pid
    return result


def load_csv_map(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["predicate_label"]: row["predicate_id"] for row in reader if row.get("predicate_id")}


def load_entities_map(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pid = entry.get("id", "")
            if not pid.startswith("P"):
                continue
            name = entry.get("name", "")
            if name:
                result[name] = pid
    return result


def wikidata_lookup(label):
    params = urllib.parse.urlencode({
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "type": "property",
        "format": "json",
        "limit": 50,
    })
    req = urllib.request.Request(
        f"{wikidata_api}?{params}",
        headers={"User-Agent": "bachelorthesis-re/1.0 (ge94say@mytum.de)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    for result in data.get("search", []):
        if result.get("label", "").lower() == label.lower():
            return result["id"]
        match = result.get("match", {})
        if match.get("type") == "alias" and match.get("text", "").lower() == label.lower():
            return result["id"]
    return ""


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        relations = json.load(f)

    nonorig_map  = load_nonorig_map(all_rebel_relations)
    csv_map      = load_csv_map(relations_yuchen)
    entities_map = load_entities_map(entities_jsonl) if args.dataset == "tekgen" else {}

    n_manual = n_nonorig = n_csv = n_entities = n_api = n_missing = 0

    for i, entry in enumerate(relations):
        label = entry["predicate_label"]

        # 0. entities.jsonl (tekgen only, first priority)
        if entities_map:
            pid = entities_map.get(label, "")
            if pid:
                entry["predicate_id"] = pid
                n_entities += 1
                continue

        # 1. manual map
        pid = manual_map.get(label, "")
        if pid:
            entry["predicate_id"] = pid
            n_manual += 1
            continue

        # 2. nonorig_relations.json
        pid = nonorig_map.get(label, "")
        if pid:
            entry["predicate_id"] = pid
            n_nonorig += 1
            continue

        # 3. predicate_descriptions.csv
        pid = csv_map.get(label, "")
        if pid:
            entry["predicate_id"] = pid
            n_csv += 1
            continue

        # 4. Wikidata API
        print(f"  [{i+1}/{len(relations)}] API lookup: {label!r}")
        try:
            pid = wikidata_lookup(label)
            time.sleep(API_DELAY)
        except Exception as e:
            print(f"    ERROR: {e}")
            pid = ""

        entry["predicate_id"] = pid
        if pid:
            n_api += 1
            print(f"    -> {pid}")
        else:
            n_missing += 1
            print(f"    -> not found")

    print(f"\nResults for {len(relations)} relations:")
    if args.dataset == "tekgen":
        print(f"  From entities.jsonl              : {n_entities}")
    print(f"  From manual map                  : {n_manual}")
    print(f"  From nonorig_relations.json      : {n_nonorig}")
    print(f"  From predicate_descriptions.csv  : {n_csv}")
    print(f"  From Wikidata API                : {n_api}")
    print(f"  Still missing                    : {n_missing}")

    if n_missing:
        print("\nMissing predicate_ids:")
        for e in relations:
            if not e.get("predicate_id"):
                print(f"  {e.get('predicate_label')!r}")

    found   = [e for e in relations if e.get("predicate_id")]
    missing = [e for e in relations if not e.get("predicate_id")]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(found)} found    -> {OUTPUT_FILE}")

    with open(OUTPUT_FILE_MISSING, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(missing)} missing -> {OUTPUT_FILE_MISSING}")


if __name__ == "__main__":
    main()
