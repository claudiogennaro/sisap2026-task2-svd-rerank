import argparse
import glob
import json
from pathlib import Path

import h5py

from task2_bench import run_svd_rerank

PREFERRED_BASE_NAMES = ["train", "learn", "base", "database", "vectors", "xb"]
PREFERRED_QUERY_NAMES = ["test", "queries", "query", "xq", "dev", "eval"]
EXCLUDED_DATASET_TOKENS = {
    "neighbor",
    "distance",
    "dist",
    "knn",
    "groundtruth",
    "ground_truth",
    "label",
    "id",
    "idx",
}


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


def resolve_input_h5s(args):
    if args.input_h5:
        return [args.input_h5]
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            matches = sorted(str(path) for path in input_path.rglob("*.h5"))
            if matches:
                return matches
        matches = sorted(glob.glob(args.input))
        if matches:
            return matches
        matches = sorted(glob.glob(args.input, recursive=True))
        if matches:
            return matches
        if input_path.exists():
            if input_path.is_file():
                return [args.input]
            matches = sorted(str(path) for path in input_path.rglob("*.h5"))
            if matches:
                return matches
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


def is_excluded_dataset(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in EXCLUDED_DATASET_TOKENS)


def collect_vector_candidates(h5_paths):
    candidates = []
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as handle:
            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if len(obj.shape) != 2:
                    return
                if obj.dtype.kind != "f":
                    return
                if is_excluded_dataset(name):
                    return
                candidates.append(
                    {
                        "path": h5_path,
                        "name": name,
                        "rows": int(obj.shape[0]),
                        "dim": int(obj.shape[1]),
                    }
                )
            handle.visititems(visit)
    if not candidates:
        raise SystemExit(f"No floating-point 2D vector datasets found in: {h5_paths}")
    return candidates


def choose_common_dim(candidates):
    dim_stats = {}
    for candidate in candidates:
        stats = dim_stats.setdefault(candidate["dim"], {"count": 0, "rows": 0})
        stats["count"] += 1
        stats["rows"] += candidate["rows"]
    return max(dim_stats.items(), key=lambda item: (item[1]["count"], item[1]["rows"], item[0]))[0]


def pick_named_candidate(candidates, preferred_names):
    by_name = {candidate["name"]: candidate for candidate in candidates}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]
    return None


def find_candidate(candidates, dataset_name):
    for candidate in candidates:
        if candidate["name"] == dataset_name:
            return candidate
    return None


def infer_inputs(h5_paths, train_dset: str, query_dset: str, task_description=None):
    candidates = collect_vector_candidates(h5_paths)
    common_dim = choose_common_dim(candidates)
    candidates = [candidate for candidate in candidates if candidate["dim"] == common_dim]

    if task_description:
        configured_base = task_description.get("data")
        configured_query = task_description.get("queries")
        if isinstance(configured_base, str) and isinstance(configured_query, str):
            base_candidate = find_candidate(candidates, configured_base)
            query_candidate = find_candidate(candidates, configured_query)
            if base_candidate and query_candidate:
                return base_candidate, query_candidate

    by_file = {}
    for candidate in candidates:
        by_file.setdefault(candidate["path"], []).append(candidate)

    for file_candidates in by_file.values():
        names = {candidate["name"] for candidate in file_candidates}
        if train_dset in names and query_dset in names:
            return (
                next(candidate for candidate in file_candidates if candidate["name"] == train_dset),
                next(candidate for candidate in file_candidates if candidate["name"] == query_dset),
            )

    for file_candidates in by_file.values():
        base_candidate = pick_named_candidate(file_candidates, PREFERRED_BASE_NAMES)
        query_candidate = pick_named_candidate(file_candidates, PREFERRED_QUERY_NAMES)
        if base_candidate and query_candidate and base_candidate["name"] != query_candidate["name"]:
            return base_candidate, query_candidate

    base_candidate = pick_named_candidate(candidates, PREFERRED_BASE_NAMES)
    query_candidate = pick_named_candidate(candidates, PREFERRED_QUERY_NAMES)
    if base_candidate and query_candidate and (
        base_candidate["path"] != query_candidate["path"] or base_candidate["name"] != query_candidate["name"]
    ):
        return base_candidate, query_candidate

    sorted_candidates = sorted(candidates, key=lambda candidate: (candidate["rows"], candidate["name"], candidate["path"]))
    query_candidate = sorted_candidates[0]
    base_candidate = sorted(candidates, key=lambda candidate: (candidate["rows"], candidate["name"], candidate["path"]))[-1]
    if len(candidates) == 1:
        return base_candidate, base_candidate
    if base_candidate["path"] == query_candidate["path"] and base_candidate["name"] == query_candidate["name"]:
        raise SystemExit(
            "Unable to infer distinct base/query datasets from the provided HDF5 inputs. "
            f"Candidates: {[(c['path'], c['name'], c['rows'], c['dim']) for c in candidates]}"
        )
    return base_candidate, query_candidate


def infer_result_dataset_name(task_description, base_candidate, query_candidate):
    if task_description:
        for key in ("dataset_name", "dataset", "dataset_id", "input", "inputDataset"):
            value = task_description.get(key)
            if isinstance(value, str) and value:
                return Path(value).stem
    if query_candidate:
        return Path(query_candidate["path"]).stem
    return Path(base_candidate["path"]).stem


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

    input_h5s = resolve_input_h5s(args)
    base_candidate, query_candidate = infer_inputs(input_h5s, args.train_dset, args.query_dset, task_description)
    print(f"Using base file: {base_candidate['path']}")
    print(f"Using base dataset: {base_candidate['name']}")
    print(f"Using query file: {query_candidate['path']}")
    print(f"Using query dataset: {query_candidate['name']}")
    output_h5 = resolve_output_h5(args)
    dataset_name = infer_result_dataset_name(task_description, base_candidate, query_candidate)

    bench_args = argparse.Namespace(
        base_h5=base_candidate["path"],
        query_h5=query_candidate["path"],
        base_dset=base_candidate["name"],
        query_dset=query_candidate["name"],
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
        dataset_name=dataset_name,
    )
    run_svd_rerank(bench_args)


if __name__ == "__main__":
    main()
