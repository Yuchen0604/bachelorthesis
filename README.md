
### REBEL
Download the sentence-level dataset from [Zenodo](https://zenodo.org/records/6139236) and place the files into `data_rebel/rebel/`:
```
data_rebel/rebel/en_train.jsonl
data_rebel/rebel/en_val.jsonl
data_rebel/rebel/en_test.jsonl
```

### Lagrange
Download from (https://aclanthology.org/2024.lrec-main.335/). Place the files into `data_lagrange/lagrange/`:
```
data_lagrange/lagrange/lagrange_train.json
data_lagrange/lagrange/lagrange_test.json
```

### TekGen
Download from (https://github.com/google-research-datasets/KELM-corpus) and place into `data_tekgen/tekgen/`:
```
data_tekgen/tekgen/quadruples-train.tsv
data_tekgen/tekgen/quadruples-validation.tsv
data_tekgen/tekgen/quadruples-test.tsv
```
Also download `entities.jsonl` to the repo root (used for property ID lookup).

---

## Pipeline

Replace `--dataset rebel` with `--dataset lagrange` or `--dataset tekgen` to run for other datasets.

### 1. Collect relations and build top-220 list for REBEL only
```bash
python embed_relations/collect_relations_nonorig.py        # REBEL
python embed_relations/collect_relations_lagrange.py       # Lagrange
python embed_relations/collect_relations_tekgen.py         # TekGen
```

### 2. Look up Wikidata property IDs
```bash
python embed_relations/get_predicate_id.py --dataset lagrange
python embed_relations/get_predicate_id.py --dataset tekgen
```
Not needed for REBEL (IDs already present).

### 3. Fetch Wikidata descriptions
```bash
python embed_relations/fetch_wikidata_descriptions.py --dataset rebel
python embed_relations/fetch_wikidata_descriptions.py --dataset lagrange
python embed_relations/fetch_wikidata_descriptions.py --dataset tekgen
```

### 4. Embed relations with SBERT
```bash
python embed_relations/embed_relations.py --dataset rebel
```
Repeat with `--dataset lagrange` and `--dataset tekgen`.

### 5. Convert raw data to standardized format
```bash
python conversion/BA_convert_rebel.py
python conversion/BA_convert_lagrange.py
python conversion/BA_convert_tekgen.py
```
Outputs: `data_{dataset}/re_data/{dataset}_{train,val,test}.jsonl`

### 6. Build hard-negative candidates
```bash
python embed_relations/find_candidates.py --dataset rebel
```
Repeat with `--dataset lagrange` and `--dataset tekgen`.

### 7. Build instruction format
```bash
python instruction/BG_build_instructions.py --dataset rebel
```
Repeat with `--dataset lagrange` and `--dataset tekgen`.
Outputs: `data_{dataset}/dataset-instruct/{train,valid,test}.jsonl`

### 8. Downsample to 20K
```bash
python finetuning/downsample.py --dataset rebel
```
Repeat with `--dataset lagrange` and `--dataset tekgen`.
Outputs: `data_{dataset}/dataset-instruct-20k/{train,valid,test}.jsonl` (16K train / 2K valid / 2K test)

### 9. Fine-tune
```bash
sbatch finetuning/train.sbatch
```
Edit `RUN`, `DATASET`, and model list in the script before submitting.

### 10. Run inference
```bash
sbatch evaluation/evaluate_all_epochs.sbatch
```

### 11. Evaluate predictions
```bash
python evaluation/evaluate_checkpoints.py --run run2 --dataset rebel
```
Logs epoch-level metrics to W&B.

---

## Directory Structure

```
conversion/          — raw-to-standardized converters (BA_convert_*.py)
instruction/         — instruction format builder (BG_build_instructions.py)
embed_relations/     — relation collection, ID lookup, SBERT embedding, candidate sampling
  relations_{dataset}/
    {dataset}_relations.json             — all unique relations found in the dataset (e.g. 1166 for Lagrange)
    {dataset}_relations_missing.json     — relations with no Wikidata ID match (e.g. 40 for Lagrange);
                                           excluded from embedding and candidate sampling
    {dataset}_relations_predicate_id.json — mapping of relation labels to Wikidata  IDs
    {dataset}_relations_wikidata.json    — relations with a resolved ID and Wikidata description
                                           (e.g. 1126 for Lagrange)
                                           total = missing (40) + wikidata (1126) = 1166
finetuning/          — fine-tuning and downsampling
evaluation/          — inference, evaluation, W&B logging
data_rebel/          — REBEL dataset pipeline
data_lagrange/       — Lagrange dataset pipeline
data_tekgen/         — TekGen dataset pipeline
predictions/         — model inference outputs
evaluation_results/  — per-sample and summary evaluation JSONs
