import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


RUNS = [
    {
        "name": "svd_d76_m10_bs256",
        "args": [
            "src/task2_bench.py",
            "svd-rerank",
            "--base-h5", "data/llama-128-ip.hdf5",
            "--query-h5", "data/llama-128-ip.hdf5",
            "--base-dset", "train",
            "--query-dset", "test",
            "--d", "76",
            "--m", "10",
            "--batch-size", "256",
            "--topk", "30",
            "--gt-npy", "runs/exact_ids.npy",
            "--output", "runs/svd_d76_m10_bs256.json",
        ],
    },
    {
        "name": "svd_d76_m10_bs1024",
        "args": [
            "src/task2_bench.py",
            "svd-rerank",
            "--base-h5", "data/llama-128-ip.hdf5",
            "--query-h5", "data/llama-128-ip.hdf5",
            "--base-dset", "train",
            "--query-dset", "test",
            "--d", "76",
            "--m", "10",
            "--batch-size", "1024",
            "--topk", "30",
            "--gt-npy", "runs/exact_ids.npy",
            "--output", "runs/svd_d76_m10_bs1024.json",
        ],
    },
    {
        "name": "svd_d76_m10_fp16",
        "args": [
            "src/task2_bench.py",
            "svd-rerank",
            "--base-h5", "data/llama-128-ip.hdf5",
            "--query-h5", "data/llama-128-ip.hdf5",
            "--base-dset", "train",
            "--query-dset", "test",
            "--d", "76",
            "--m", "10",
            "--batch-size", "512",
            "--topk", "30",
            "--compact-fp16",
            "--gt-npy", "runs/exact_ids.npy",
            "--output", "runs/svd_d76_m10_fp16.json",
        ],
    },
    {
        "name": "svd_d76_m10_bs1024_fp16",
        "args": [
            "src/task2_bench.py",
            "svd-rerank",
            "--base-h5", "data/llama-128-ip.hdf5",
            "--query-h5", "data/llama-128-ip.hdf5",
            "--base-dset", "train",
            "--query-dset", "test",
            "--d", "76",
            "--m", "10",
            "--batch-size", "1024",
            "--topk", "30",
            "--compact-fp16",
            "--gt-npy", "runs/exact_ids.npy",
            "--output", "runs/svd_d76_m10_bs1024_fp16.json",
        ],
    },
]


def run_one(run):
    print(f"\n===== Running {run['name']} =====")
    cmd = [PYTHON, *run["args"]]
    subprocess.run(cmd, cwd=ROOT, check=True)
    output_path = ROOT / next(arg for i, arg in enumerate(run["args"]) if run["args"][i - 1] == "--output")
    return json.loads(output_path.read_text())


def main():
    results = [run_one(run) for run in RUNS]

    print("\n===== Summary =====")
    for result in results:
        print(
            f"{result['method']} "
            f"d={result['proj_dim']} m={result['m']} "
            f"bs={result['batch_size']} fp16={result.get('compact_fp16', False)} "
            f"search_time_s={result['search_time_s']:.6f} "
            f"qps={result['qps']:.2f} "
            f"recall={result.get('final_recall_at_k', float('nan')):.6f}"
        )


if __name__ == "__main__":
    main()
