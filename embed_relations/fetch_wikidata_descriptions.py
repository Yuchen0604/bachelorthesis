import argparse
import json
import os
import time
import urllib.request
import urllib.parse

base_dir = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    "rebel": {
        "in_path":  os.path.join(base_dir, "relations_rebel",   "220_nonorig_relations.json"),
        "out_path": os.path.join(base_dir, "relations_rebel",   "220_nonorig_relations_wikidata.json"),
    },
    "rebel-full": {
        "in_path":  os.path.join(base_dir, "relations_rebel",   "nonorig_relations.json"),
        "out_path": os.path.join(base_dir, "relations_rebel",   "nonorig_relations_wikidata.json"),
    },
    "lagrange": {
        "in_path":  os.path.join(base_dir, "relations_lagrange", "lagrange_relations_predicate_id.json"),
        "out_path": os.path.join(base_dir, "relations_lagrange", "lagrange_relations_wikidata.json"),
    },
    "tekgen": {
        "in_path":  os.path.join(base_dir, "relations_tekgen", "tekgen_relations_predicate_id.json"),
        "out_path": os.path.join(base_dir, "relations_tekgen", "tekgen_relations_wikidata.json"),
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["rebel", "rebel-full", "lagrange", "tekgen"], default="rebel")
args = parser.parse_args()

in_path  = DATASET_CONFIG[args.dataset]["in_path"]
out_path = DATASET_CONFIG[args.dataset]["out_path"]

wikidata_api ="https://www.wikidata.org/w/api.php"
batch_size   = 50

#fetch wikidata description for top 220 relations (for embeddings needed)
def fetch_descriptions(pids: list[str]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for i in range(0, len(pids), batch_size):
        batch = pids[i:i + batch_size]
        params = urllib.parse.urlencode({
            "action":    "wbgetentities",
            "ids":       "|".join(batch),
            "props":     "descriptions",
            "languages": "en",
            "format":    "json",
        })
        req = urllib.request.Request(
            f"{wikidata_api}?{params}",
            headers={"User-Agent": "bachelorthesis-re/1.0 (ge94say@mytum.de)"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = int(e.headers.get("Retry-After", 2 ** attempt * 2))
                    print(f"  rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        for pid, entity in result.get("entities", {}).items():
            desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
            descriptions[pid] = desc
        print(f"  {min(i + batch_size, len(pids))}/{len(pids)} fetched")
        if i + batch_size < len(pids):
            time.sleep(1.0)
    return descriptions


def main():
    with open(in_path, encoding="utf-8") as f:
        relations = json.load(f)

    pids = [r["predicate_id"] for r in relations if r.get("predicate_id")]
    skipped = len(relations) - len(pids)
    print(f"Fetching descriptions for {len(pids)}/{len(relations)} properties ({skipped} skipped — no predicate_id)...")
    descriptions = fetch_descriptions(pids)

    for r in relations:
        r["wikidata_description"] = descriptions.get(r.get("predicate_id", ""), "")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
