import random
import argparse
import os

DATASET_CONFIG = {
    "rebel": {
        "input_dir":  "data_rebel/dataset-instruct",
        "output_dir": "data_rebel/dataset-instruct-20k",
    },
    "lagrange": {
        "input_dir":  "data_lagrange/dataset-instruct",
        "output_dir": "data_lagrange/dataset-instruct-20k",
    },
}

# rebel: read train/valid/test from separate files
REBEL_SPLITS = {
    "train": 16_000,
    "valid":  2_000,
    "test":   2_000,
}


def sample_lines(lines, n, seed):
    random.seed(seed)
    if len(lines) <= n:
        print(f"  Only {len(lines)} samples available, using all.")
        return lines
    sampled = random.sample(lines, n)
    print(f"  Sampled {n:,} from {len(lines):,}.")
    return sampled


def read_jsonl(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def write_jsonl(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def run_rebel(input_dir, output_dir, seed):
    for split, n in REBEL_SPLITS.items():
        input_path  = os.path.join(input_dir,  f"{split}.jsonl")
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        print(f"{split} ({n:,}):")
        lines = read_jsonl(input_path)
        write_jsonl(output_path, sample_lines(lines, n, seed))


def run_lagrange(input_dir, output_dir, seed):
    # test → 2k
    test_input  = os.path.join(input_dir,  "test.jsonl")
    test_output = os.path.join(output_dir, "test.jsonl")
    print("test (2,000):")
    write_jsonl(test_output, sample_lines(read_jsonl(test_input), 2_000, seed))

    #train: sample 18k, then split 16k train + 2k valid
    train_input = os.path.join(input_dir, "train.jsonl")
    print("train (18,000 → 16,000 train + 2,000 valid):")
    pool = sample_lines(read_jsonl(train_input), 18_000, seed)
    random.seed(seed)
    random.shuffle(pool)
    write_jsonl(os.path.join(output_dir, "valid.jsonl"), pool[:2_000])
    write_jsonl(os.path.join(output_dir, "train.jsonl"), pool[2_000:])
    print(f"  valid: 2,000  |  train: {len(pool[2_000:]):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["rebel", "lagrange"], default="rebel")
    parser.add_argument("--input_dir",  default=None, help="Override default input dir")
    parser.add_argument("--output_dir", default=None, help="Override default output dir")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    input_dir  = args.input_dir  or cfg["input_dir"]
    output_dir = args.output_dir or cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    if args.dataset == "rebel":
        run_rebel(input_dir, output_dir, args.seed)
    else:
        run_lagrange(input_dir, output_dir, args.seed)
