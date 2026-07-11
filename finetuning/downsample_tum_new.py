import json
import os
import random

seed   = 42
CN     = 20
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(BASE_DIR, "data_tum", "tum_p99-20k", f"tum_p99_test_cn{CN}.jsonl")
IN_FILE   = os.path.join(BASE_DIR, "data_tum", "re_data", f"wiki_s2t_p99_new_cn{CN}.jsonl")
OUT_DIR   = os.path.join(BASE_DIR, "data_tum", "tum_p99_new")

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(seed)

test_keys = set()
test_samples = []
with open(TEST_FILE, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        test_keys.add((d["sentence_id"], d["entity"]))
        test_samples.append(line.strip())
print(f"Test set: {len(test_keys):,} samples to exclude")

#filter out test samples
remaining = []
n_excluded = 0
with open(IN_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if (d["sentence_id"], d["entity"]) in test_keys:
            n_excluded += 1
        else:
            remaining.append(line)

print(f"Excluded {n_excluded:,} test samples from new dataset")
print(f"Remaining: {len(remaining):,} samples")

random.shuffle(remaining)
n_train = round(len(remaining) * TRAIN_RATIO)
n_valid = round(len(remaining) * VALID_RATIO)

splits = {
    "train": remaining[:n_train],
    "valid": remaining[n_train:n_train + n_valid],
    "test":  remaining[n_train + n_valid:],
}

for split_name, lines in splits.items():
    out_path = os.path.join(OUT_DIR, f"tum_p99_new_{split_name}_cn{CN}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"  {split_name}: {len(lines):,} → {out_path}")
