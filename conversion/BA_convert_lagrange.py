import json
import os
import re

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR    = os.path.join(ROOT_DIR, "data_lagrange", "re_data")
LAGRANGE_DIR  = os.path.join(ROOT_DIR, "data_lagrange", "lagrange")
WIKIDATA_FILE = os.path.join(ROOT_DIR, "embed_relations", "relations_lagrange", "lagrange_relations_wikidata.json")

TRIPLE_RE = re.compile(r"<S>(.*?)<P>(.*?)<O>(.*)$")


def load_allowed_relations(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        relations = json.load(f)
    return {r["predicate_label"] for r in relations}


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


def convert_records(records, output_path: str, allowed_relations: set[str]) -> int:
    written  = 0
    skipped_no_triples    = 0
    skipped_disallowed    = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, record in enumerate(records):
            triple_str = record.get("triples", "")
            if not triple_str:
                skipped_no_triples += 1
                continue

            parsed = parse_triples(triple_str)
            if not parsed:
                skipped_no_triples += 1
                continue

            # filter out entire sample if any relation is not allowed
            if any(rel not in allowed_relations for _, rel, _ in parsed):
                skipped_disallowed += 1
                continue

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
            ]

            title = record.get("title", "").replace(" ", "_")
            if not title:
                print(f"  WARNING: sample {idx} has no title, using 'sample_{idx}'")
            entity = title if title else f"sample_{idx}"

            fout.write(json.dumps({
                "category": "lagrange",
                "entity":   entity,
                "sentence": record.get("sentence", ""),
                "triples":  triples,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {written:>8,} written  →  {output_path}")
    print(f"    skipped {skipped_no_triples:>6,}  — no triples parsed")
    print(f"    skipped {skipped_disallowed:>6,}  — contained disallowed relation")
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    allowed = load_allowed_relations(WIKIDATA_FILE)
    print(f"Loaded {len(allowed)} allowed relations from {WIKIDATA_FILE}\n")

    print("Reading lagrange_test.json ...")
    test_records = read_jsonl(os.path.join(LAGRANGE_DIR, "lagrange_test.json"))
    print(f"  {len(test_records):,} records")
    print("Converting test ...")
    convert_records(test_records, os.path.join(OUTPUT_DIR, "lagrange_test.jsonl"), allowed)

    print("\nReading lagrange_train.json ...")
    train_records = read_jsonl(os.path.join(LAGRANGE_DIR, "lagrange_train.json"))
    print(f"  {len(train_records):,} records")
    print("Converting train ...")
    convert_records(train_records, os.path.join(OUTPUT_DIR, "lagrange_train.jsonl"), allowed)

    print("\nDone.")


if __name__ == "__main__":
    main()
