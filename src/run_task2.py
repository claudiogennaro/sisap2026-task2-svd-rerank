import argparse
import glob
import json
from pathlib import Path

import h5py

from task2_bench import run_svd_rerank


def build_parser():
    parser = argparse.ArgumentParser(description="Final runner for SISAP 2026 Task 2.")

    # Legacy/local interface
    parser.add_argument("--input-h5", help="Path to the HDF5 file containing train/test.")
    parser.add_argument("--train-dset", default="train", help="Base dataset name.")
    parser.add_argument("--query-dset", default="test", help="Query dataset name.")
    parser.add_argument("--output-h5", help="Output HDF5 path.")

    # TIRA-style interface
    parser.add_argument("--input", help="Input HDF5 path or glob, e.g. $inputDataset/*.h5")
    parser.add_argument("--task-description", help="Path to the TIRA task description JSON.")
    parser.add_argument("--output", help="Output directory provided by TIRA.")

    # Shared algorithm parameters
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d", type=int, default=76, help="TruncatedSVD dimension.")
    parser.add_argument("--m", type=int, default=10, help="Candidate expansion factor: candidate_k = topk * m.")
    parser.add_argument("--n-iter", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo-name", default="svd-rerank")
    parser.add_argument("--task-name", default="task2")
    return parser


def resolve_input_h5(args) -> str:
    if args.input_h5:
        return args.input_h5
    if args.input:
        matches = sorted(glob.glob(args.input))
        if matches:
            # TIRA spot-check datasets may contain multiple .h5 files.
            # Prefer the one that exposes the expected train/test datasets.
            for candidate in matches:
                try:
                    with h5py.File(candidate, "r") as handle:
                        if args.train_dset in handle and args.query_dset in handle:
                            return candidate
                except Exception:
                    continue
            return matches[0]
        if Path(args.input).exists():
            return args.input
    raise SystemExit("No input dataset found. Provide --input-h5 or --input.")


def resolve_output_h5(args) -> str:
    if args.output_h5:
        out = Path(args.output_h5)
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        # TIRA expects .h5 files directly in the provided output directory.
        return str(output_dir / f"{args.algo_name}_d{args.d}_m{args.m}.h5")

    default_out = Path("results") / args.task_name / f"{args.algo_name}_d{args.d}_m{args.m}.h5"
    default_out.parent.mkdir(parents=True, exist_ok=True)
    return str(default_out)


def maybe_load_task_description(args):
    if not args.task_description:
        return None
    path = Path(args.task_description)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    parser = build_parser()
    args = parser.parse_args()

    task_description = maybe_load_task_description(args)
    if task_description:
        args.task_name = task_description.get("task", args.task_name)

    input_h5 = resolve_input_h5(args)
    output_h5 = resolve_output_h5(args)

    bench_args = argparse.Namespace(
        base_h5=input_h5,
        query_h5=input_h5,
        base_dset=args.train_dset,
        query_dset=args.query_dset,
        d=args.d,
        m=args.m,
        topk=args.topk,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        seed=args.seed,
        compact_fp16=False,
        gt_npy=None,
        output=None,
        output_ids_npy=None,
        output_h5=output_h5,
        algo_name=args.algo_name,
        task_name=args.task_name,
    )
    run_svd_rerank(bench_args)


if __name__ == "__main__":
    main()
