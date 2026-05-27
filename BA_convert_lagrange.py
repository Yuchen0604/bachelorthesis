import json
import os
import re

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RELATIONS_JSON = os.path.join(SCRIPT_DIR, "embed_relations", "relations_lagrange", "220_lagrange_relations.json")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "data_lagrange", "re_data")
LAGRANGE_DIR   = os.path.join(SCRIPT_DIR, "data_lagrange", "lagrange")

TRIPLE_RE = re.compile(r"<S>(.*?)<P>(.*?)<O>(.*)$")


def parse_triples(triple_str: str) -> list[tuple[str, str, str]]:
    result = []
    for chunk in triple_str.split("<sep>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = TRIPLE_RE.match(chunk)
        if m:
            result.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return result


def load_allowed_relations(json_path: str) -> set[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["predicate_label"] for entry in data if entry.get("predicate_label")}


def convert_records(records, allowed_relations, output_path: str) -> int:
    written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, record in enumerate(records):
            triple_str = record.get("triples", "")
            if not triple_str:
                skipped += 1
                continue

            parsed = parse_triples(triple_str)
            triples = [
                {
                    "triple_index":    i,
                    "subject_id":      "",
                    "subject_label":   subj,
                    "predicate_id":    "",
                    "predicate_label": rel,
                    "object_id":       "",
                    "object_label":    obj,
                }
                for i, (subj, rel, obj) in enumerate(parsed)
                if rel in allowed_relations
            ]

            if not triples:
                skipped += 1
                continue

            title  = record.get("title", "").replace(" ", "_")
            entity = title if title else f"sample_{idx}"

            fout.write(json.dumps({
                "category": "lagrange",
                "entity":   entity,
                "sentence": record.get("sentence", ""),
                "triples":  triples,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {written:>8,} written  →  {output_path}")
    print(f"    skipped {skipped:>8,}  — no triples after relation filter")
    return written


def read_jsonl(path: str, max_records: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_records and len(records) >= max_records:
                break
    return records


def main():
    allowed_relations = load_allowed_relations(RELATIONS_JSON)
    print(f"Loaded {len(allowed_relations)} allowed relations\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading lagrange_test.json ...")
    test_records = read_jsonl(os.path.join(LAGRANGE_DIR, "lagrange_test.json"))
    print(f"  {len(test_records):,} records")
    print("Converting test ...")
    convert_records(test_records, allowed_relations,
                    os.path.join(OUTPUT_DIR, "lagrange_test.jsonl"))

    print("\nReading lagrange_train.json ...")
    train_records = read_jsonl(os.path.join(LAGRANGE_DIR, "lagrange_train.json"))
    print(f"  {len(train_records):,} records")
    print("Converting train ...")
    convert_records(train_records, allowed_relations,
                    os.path.join(OUTPUT_DIR, "lagrange_train.jsonl"))

    print("\nDone.")


if __name__ == "__main__":
    main()
