import argparse
from pathlib import Path

from task2_bench import run_svd_rerank


def build_parser():
    parser = argparse.ArgumentParser(description="Runner finale per SISAP 2026 Task 2.")
    parser.add_argument("--input-h5", required=True, help="Path del file HDF5 con train/test.")
    parser.add_argument("--train-dset", default="train", help="Nome dataset base.")
    parser.add_argument("--query-dset", default="test", help="Nome dataset query.")
    parser.add_argument("--output-h5", default="results/task2/svd_d76_m10.h5", help="Path output HDF5.")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d", type=int, default=76, help="Dimensione TruncatedSVD.")
    parser.add_argument("--m", type=int, default=10, help="Fattore candidati: candidate_k = topk * m.")
    parser.add_argument("--n-iter", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo-name", default="svd-rerank")
    parser.add_argument("--task-name", default="task2")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    out = Path(args.output_h5)
    out.parent.mkdir(parents=True, exist_ok=True)

    bench_args = argparse.Namespace(
        base_h5=args.input_h5,
        query_h5=args.input_h5,
        base_dset=args.train_dset,
        query_dset=args.query_dset,
        d=args.d,
        m=args.m,
        topk=args.topk,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        seed=args.seed,
        gt_npy=None,
        output=None,
        output_ids_npy=None,
        output_h5=str(out),
        algo_name=args.algo_name,
        task_name=args.task_name,
    )
    run_svd_rerank(bench_args)


if __name__ == "__main__":
    main()
