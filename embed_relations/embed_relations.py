import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

base_dir = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    "rebel": {
        "relations_path": os.path.join(base_dir, "relations_rebel", "220_nonorig_relations_wikidata.json"),
        "cache_path":     os.path.join(base_dir, "sbert_embeddings.npz"),
    },
    "lagrange": {
        "relations_path": os.path.join(base_dir, "relations_lagrange", "lagrange_relations_wikidata.json"),
        "cache_path":     os.path.join(base_dir, "sbert_embeddings_lagrange.npz"),
    },
    "tekgen": {
        "relations_path": os.path.join(base_dir, "relations_tekgen", "tekgen_relations_wikidata.json"),
        "cache_path":     os.path.join(base_dir, "sbert_embeddings_tekgen.npz"),
    },
    "tum": {
        "relations_path": os.path.join(base_dir, "relations_tum", "tum_relations_wikidata.json"),
        "cache_path":     os.path.join(base_dir, "sbert_embeddings_tum.npz"),
    },
    "tum_p99_new": {
        "relations_path": os.path.join(base_dir, "relations_tum", "tum_relations_wikidata_p99_new.json"),
        "cache_path":     os.path.join(base_dir, "sbert_embeddings_tum_p99_new.npz"),
    },
    "wikinre": {
        "relations_path": os.path.join(base_dir, "relations_wikinre", "wikinre_relations_wikidata.json"),
        "cache_path":     os.path.join(base_dir, "sbert_embeddings_wikinre.npz"),
    },
}


def load_relations(relations_path: str) -> tuple[list[str], list[str]]:
    with open(relations_path, encoding="utf-8") as f:
        data = json.load(f)
    relations = sorted(entry["predicate_label"] for entry in data)
    label_to_desc = {entry["predicate_label"]: entry.get("wikidata_description", "") for entry in data}
    texts = [
        f"{r}: {label_to_desc[r]}" if label_to_desc.get(r) else r
        for r in relations
    ]
    return relations, texts


def embed(relations: list[str], texts: list[str], cache_path: str) -> np.ndarray:
    print("Encoding relations with SBERT (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.savez(cache_path, relations=relations, embeddings=embeddings)
    print(f"Saved {len(relations)} embeddings to {cache_path}")
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rebel", "lagrange", "tekgen", "tum", "tum_p99_new", "wikinre"])
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    relations, texts = load_relations(cfg["relations_path"])
    print(f"Loaded {len(relations)} relations from {cfg['relations_path']}")
    embed(relations, texts, cfg["cache_path"])


if __name__ == "__main__":
    main()
