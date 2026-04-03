import argparse
import json
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import psutil
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss-cpu non installato. Esegui: pip install -r requirements.txt") from exc


def inspect_h5(path: str) -> None:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"GROUP   {name}")

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)


def load_array(h5_path: str, dataset_name: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"Dataset '{dataset_name}' non trovato in {h5_path}")
        data = np.asarray(handle[dataset_name], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Atteso array 2D, trovato shape={data.shape}")
    return np.ascontiguousarray(data)


def batched_search(index, queries: np.ndarray, topk: int, batch_size: int):
    all_scores = []
    all_ids = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        scores, ids_ = index.search(batch, topk)
        all_scores.append(scores)
        all_ids.append(ids_)
    return np.vstack(all_scores), np.vstack(all_ids)


def mean_recall_at_k(found: np.ndarray, gt: np.ndarray, topk: int) -> float:
    recalls = []
    for row_found, row_gt in zip(found[:, :topk], gt[:, :topk]):
        recalls.append(len(set(row_found.tolist()) & set(row_gt.tolist())) / topk)
    return float(np.mean(recalls))


def candidate_recall(found: np.ndarray, gt: np.ndarray, topk: int) -> float:
    recalls = []
    for row_found, row_gt in zip(found, gt[:, :topk]):
        recalls.append(len(set(row_found.tolist()) & set(row_gt.tolist())) / topk)
    return float(np.mean(recalls))


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


def write_result_h5(
    path: str,
    ids_: np.ndarray,
    scores: np.ndarray,
    algo: str,
    task: str,
    build_time_s: float,
    search_time_s: float,
    params: dict,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as handle:
        # Challenge files use 1-based identifiers.
        handle.create_dataset("knns", data=np.asarray(ids_, dtype=np.int64) + 1)
        handle.create_dataset("dists", data=np.asarray(scores, dtype=np.float32))
        handle.attrs["algo"] = algo
        handle.attrs["task"] = task
        handle.attrs["buildtime"] = float(build_time_s)
        handle.attrs["querytime"] = float(search_time_s)
        handle.attrs["params"] = json.dumps(params, sort_keys=True)


def run_exact(args) -> None:
    base = load_array(args.base_h5, args.base_dset)
    queries = load_array(args.query_h5, args.query_dset)

    build_start = time.perf_counter()
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    build_time = time.perf_counter() - build_start

    # Warm-up on a tiny batch to avoid measuring first-touch costs.
    warm_n = min(len(queries), max(1, min(args.batch_size, 8)))
    if warm_n:
        index.search(queries[:warm_n], args.topk)

    search_start = time.perf_counter()
    scores, ids_ = batched_search(index, queries, args.topk, args.batch_size)
    search_time = time.perf_counter() - search_start

    results = {
        "method": "exact-flatip",
        "n_base": int(base.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dim": int(base.shape[1]),
        "topk": int(args.topk),
        "batch_size": int(args.batch_size),
        "build_time_s": build_time,
        "search_time_s": search_time,
        "qps": len(queries) / search_time if search_time else None,
        "rss_gb": rss_gb(),
    }

    if args.output_ids_npy:
        out = Path(args.output_ids_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, ids_)

    if args.output_h5:
        write_result_h5(
            args.output_h5,
            ids_,
            scores,
            algo=args.algo_name or "exact-flatip",
            task=args.task_name,
            build_time_s=build_time,
            search_time_s=search_time,
            params={"topk": int(args.topk), "batch_size": int(args.batch_size)},
        )

    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, indent=2))


def run_pca_rerank(args) -> None:
    base = load_array(args.base_h5, args.base_dset)
    queries = load_array(args.query_h5, args.query_dset)
    candidate_k = args.topk * args.m

    build_start = time.perf_counter()
    pca = PCA(n_components=args.d, whiten=args.whiten, svd_solver="randomized", random_state=args.seed)
    base_proj = pca.fit_transform(base).astype(np.float32, copy=False)
    index = faiss.IndexFlatIP(args.d)
    index.add(np.ascontiguousarray(base_proj))
    build_time = time.perf_counter() - build_start

    warm_n = min(len(queries), max(1, min(args.batch_size, 8)))
    if warm_n:
        warm_proj = pca.transform(queries[:warm_n]).astype(np.float32, copy=False)
        index.search(np.ascontiguousarray(warm_proj), candidate_k)

    all_final_ids = []
    all_final_scores = []
    all_coarse_ids = []

    search_start = time.perf_counter()
    for start in range(0, len(queries), args.batch_size):
        q_batch = queries[start : start + args.batch_size]
        q_proj = pca.transform(q_batch).astype(np.float32, copy=False)
        coarse_scores, coarse_ids = index.search(np.ascontiguousarray(q_proj), candidate_k)
        all_coarse_ids.append(coarse_ids)

        final_ids = np.empty((len(q_batch), args.topk), dtype=np.int64)
        final_scores = np.empty((len(q_batch), args.topk), dtype=np.float32)

        for i in range(len(q_batch)):
            cand_ids = coarse_ids[i]
            cand_vecs = base[cand_ids]
            exact_scores = cand_vecs @ q_batch[i]
            order = np.argsort(-exact_scores)[: args.topk]
            final_scores[i] = exact_scores[order]
            final_ids[i] = cand_ids[order]

        all_final_scores.append(final_scores)
        all_final_ids.append(final_ids)

    search_time = time.perf_counter() - search_start

    final_scores = np.vstack(all_final_scores)
    final_ids = np.vstack(all_final_ids)
    coarse_ids = np.vstack(all_coarse_ids)

    results = {
        "method": "pca-rerank",
        "n_base": int(base.shape[0]),
        "n_queries": int(queries.shape[0]),
        "orig_dim": int(base.shape[1]),
        "proj_dim": int(args.d),
        "topk": int(args.topk),
        "m": int(args.m),
        "candidate_k": int(candidate_k),
        "batch_size": int(args.batch_size),
        "whiten": bool(args.whiten),
        "seed": int(args.seed),
        "build_time_s": build_time,
        "search_time_s": search_time,
        "qps": len(queries) / search_time if search_time else None,
        "rss_gb": rss_gb(),
    }

    if args.gt_npy:
        gt = np.load(args.gt_npy)
        if gt.shape[0] != final_ids.shape[0]:
            raise ValueError(f"GT shape mismatch: gt={gt.shape}, result={final_ids.shape}")
        results["coarse_recall_at_c"] = candidate_recall(coarse_ids, gt, args.topk)
        results["final_recall_at_k"] = mean_recall_at_k(final_ids, gt, args.topk)

    if args.output_ids_npy:
        out = Path(args.output_ids_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, final_ids)

    if args.output_h5:
        write_result_h5(
            args.output_h5,
            final_ids,
            final_scores,
            algo=args.algo_name or "pca-rerank",
            task=args.task_name,
            build_time_s=build_time,
            search_time_s=search_time,
            params={
                "proj_dim": int(args.d),
                "m": int(args.m),
                "topk": int(args.topk),
                "batch_size": int(args.batch_size),
                "whiten": bool(args.whiten),
                "seed": int(args.seed),
            },
        )

    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, indent=2))


def run_svd_rerank(args) -> None:
    base = load_array(args.base_h5, args.base_dset)
    queries = load_array(args.query_h5, args.query_dset)
    candidate_k = args.topk * args.m

    build_start = time.perf_counter()
    svd = TruncatedSVD(n_components=args.d, algorithm="randomized", n_iter=args.n_iter, random_state=args.seed)
    proj_dtype = np.float16 if args.compact_fp16 else np.float32
    base_proj = svd.fit_transform(base).astype(proj_dtype, copy=False)
    index = faiss.IndexFlatIP(args.d)
    index.add(np.ascontiguousarray(base_proj.astype(np.float32, copy=False)))
    build_time = time.perf_counter() - build_start

    warm_n = min(len(queries), max(1, min(args.batch_size, 8)))
    if warm_n:
        warm_proj = svd.transform(queries[:warm_n]).astype(proj_dtype, copy=False)
        index.search(np.ascontiguousarray(warm_proj.astype(np.float32, copy=False)), candidate_k)

    all_final_ids = []
    all_final_scores = []
    all_coarse_ids = []

    search_start = time.perf_counter()
    for start in range(0, len(queries), args.batch_size):
        q_batch = queries[start : start + args.batch_size]
        q_proj = svd.transform(q_batch).astype(proj_dtype, copy=False)
        _, coarse_ids = index.search(np.ascontiguousarray(q_proj.astype(np.float32, copy=False)), candidate_k)
        all_coarse_ids.append(coarse_ids)

        final_ids = np.empty((len(q_batch), args.topk), dtype=np.int64)
        final_scores = np.empty((len(q_batch), args.topk), dtype=np.float32)
        for i in range(len(q_batch)):
            cand_ids = coarse_ids[i]
            cand_vecs = base[cand_ids]
            exact_scores = cand_vecs @ q_batch[i]
            order = np.argsort(-exact_scores)[: args.topk]
            final_ids[i] = cand_ids[order]
            final_scores[i] = exact_scores[order]

        all_final_ids.append(final_ids)
        all_final_scores.append(final_scores)

    search_time = time.perf_counter() - search_start

    final_ids = np.vstack(all_final_ids)
    final_scores = np.vstack(all_final_scores)
    coarse_ids = np.vstack(all_coarse_ids)

    results = {
        "method": "svd-rerank",
        "n_base": int(base.shape[0]),
        "n_queries": int(queries.shape[0]),
        "orig_dim": int(base.shape[1]),
        "proj_dim": int(args.d),
        "topk": int(args.topk),
        "m": int(args.m),
        "candidate_k": int(candidate_k),
        "batch_size": int(args.batch_size),
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "compact_fp16": bool(args.compact_fp16),
        "build_time_s": build_time,
        "search_time_s": search_time,
        "qps": len(queries) / search_time if search_time else None,
        "rss_gb": rss_gb(),
    }

    if args.gt_npy:
        gt = np.load(args.gt_npy)
        if gt.shape[0] != final_ids.shape[0]:
            raise ValueError(f"GT shape mismatch: gt={gt.shape}, result={final_ids.shape}")
        results["coarse_recall_at_c"] = candidate_recall(coarse_ids, gt, args.topk)
        results["final_recall_at_k"] = mean_recall_at_k(final_ids, gt, args.topk)

    if args.output_ids_npy:
        out = Path(args.output_ids_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, final_ids)

    if args.output_h5:
        write_result_h5(
            args.output_h5,
            final_ids,
            final_scores,
            algo=args.algo_name or "svd-rerank",
            task=args.task_name,
            build_time_s=build_time,
            search_time_s=search_time,
            params={
                "proj_dim": int(args.d),
                "m": int(args.m),
                "topk": int(args.topk),
                "batch_size": int(args.batch_size),
                "n_iter": int(args.n_iter),
                "seed": int(args.seed),
                "compact_fp16": bool(args.compact_fp16),
            },
        )

    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, indent=2))


def run_hnsw_ip(args) -> None:
    base = load_array(args.base_h5, args.base_dset)
    queries = load_array(args.query_h5, args.query_dset)

    build_start = time.perf_counter()
    index = faiss.IndexHNSWFlat(base.shape[1], args.m_hnsw, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
    index.add(base)
    build_time = time.perf_counter() - build_start

    warm_n = min(len(queries), max(1, min(args.batch_size, 8)))
    if warm_n:
        index.hnsw.efSearch = args.ef_search
        index.search(queries[:warm_n], args.topk)

    search_start = time.perf_counter()
    index.hnsw.efSearch = args.ef_search
    scores, ids_ = batched_search(index, queries, args.topk, args.batch_size)
    search_time = time.perf_counter() - search_start

    results = {
        "method": "hnsw-ip",
        "n_base": int(base.shape[0]),
        "n_queries": int(queries.shape[0]),
        "dim": int(base.shape[1]),
        "topk": int(args.topk),
        "batch_size": int(args.batch_size),
        "m_hnsw": int(args.m_hnsw),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
        "build_time_s": build_time,
        "search_time_s": search_time,
        "qps": len(queries) / search_time if search_time else None,
        "rss_gb": rss_gb(),
    }

    if args.gt_npy:
        gt = np.load(args.gt_npy)
        if gt.shape[0] != ids_.shape[0]:
            raise ValueError(f"GT shape mismatch: gt={gt.shape}, result={ids_.shape}")
        results["final_recall_at_k"] = mean_recall_at_k(ids_, gt, args.topk)

    if args.output_ids_npy:
        out = Path(args.output_ids_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, ids_)

    if args.output_h5:
        write_result_h5(
            args.output_h5,
            ids_,
            scores,
            algo=args.algo_name or "hnsw-ip",
            task=args.task_name,
            build_time_s=build_time,
            search_time_s=search_time,
            params={
                "topk": int(args.topk),
                "batch_size": int(args.batch_size),
                "m_hnsw": int(args.m_hnsw),
                "ef_construction": int(args.ef_construction),
                "ef_search": int(args.ef_search),
            },
        )

    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, indent=2))


def run_svd_hnsw_rerank(args) -> None:
    base = load_array(args.base_h5, args.base_dset)
    queries = load_array(args.query_h5, args.query_dset)
    candidate_k = args.topk * args.m

    build_start = time.perf_counter()
    svd = TruncatedSVD(n_components=args.d, algorithm="randomized", n_iter=args.n_iter, random_state=args.seed)
    proj_dtype = np.float16 if args.compact_fp16 else np.float32
    base_proj = svd.fit_transform(base).astype(proj_dtype, copy=False)
    index = faiss.IndexHNSWFlat(args.d, args.m_hnsw, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
    index.add(np.ascontiguousarray(base_proj.astype(np.float32, copy=False)))
    build_time = time.perf_counter() - build_start

    warm_n = min(len(queries), max(1, min(args.batch_size, 8)))
    if warm_n:
        warm_proj = svd.transform(queries[:warm_n]).astype(proj_dtype, copy=False)
        index.hnsw.efSearch = max(args.ef_search, candidate_k)
        index.search(np.ascontiguousarray(warm_proj.astype(np.float32, copy=False)), candidate_k)

    all_final_ids = []
    all_final_scores = []
    all_coarse_ids = []

    search_start = time.perf_counter()
    index.hnsw.efSearch = max(args.ef_search, candidate_k)
    for start in range(0, len(queries), args.batch_size):
        q_batch = queries[start : start + args.batch_size]
        q_proj = svd.transform(q_batch).astype(proj_dtype, copy=False)
        _, coarse_ids = index.search(np.ascontiguousarray(q_proj.astype(np.float32, copy=False)), candidate_k)
        all_coarse_ids.append(coarse_ids)

        final_ids = np.empty((len(q_batch), args.topk), dtype=np.int64)
        final_scores = np.empty((len(q_batch), args.topk), dtype=np.float32)
        for i in range(len(q_batch)):
            cand_ids = coarse_ids[i]
            cand_vecs = base[cand_ids]
            exact_scores = cand_vecs @ q_batch[i]
            order = np.argsort(-exact_scores)[: args.topk]
            final_ids[i] = cand_ids[order]
            final_scores[i] = exact_scores[order]

        all_final_ids.append(final_ids)
        all_final_scores.append(final_scores)

    search_time = time.perf_counter() - search_start

    final_ids = np.vstack(all_final_ids)
    final_scores = np.vstack(all_final_scores)
    coarse_ids = np.vstack(all_coarse_ids)

    results = {
        "method": "svd-hnsw-rerank",
        "n_base": int(base.shape[0]),
        "n_queries": int(queries.shape[0]),
        "orig_dim": int(base.shape[1]),
        "proj_dim": int(args.d),
        "topk": int(args.topk),
        "m": int(args.m),
        "candidate_k": int(candidate_k),
        "batch_size": int(args.batch_size),
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "compact_fp16": bool(args.compact_fp16),
        "m_hnsw": int(args.m_hnsw),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
        "build_time_s": build_time,
        "search_time_s": search_time,
        "qps": len(queries) / search_time if search_time else None,
        "rss_gb": rss_gb(),
    }

    if args.gt_npy:
        gt = np.load(args.gt_npy)
        if gt.shape[0] != final_ids.shape[0]:
            raise ValueError(f"GT shape mismatch: gt={gt.shape}, result={final_ids.shape}")
        results["coarse_recall_at_c"] = candidate_recall(coarse_ids, gt, args.topk)
        results["final_recall_at_k"] = mean_recall_at_k(final_ids, gt, args.topk)

    if args.output_ids_npy:
        out = Path(args.output_ids_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, final_ids)

    if args.output_h5:
        write_result_h5(
            args.output_h5,
            final_ids,
            final_scores,
            algo=args.algo_name or "svd-hnsw-rerank",
            task=args.task_name,
            build_time_s=build_time,
            search_time_s=search_time,
            params={
                "proj_dim": int(args.d),
                "m": int(args.m),
                "topk": int(args.topk),
                "batch_size": int(args.batch_size),
                "n_iter": int(args.n_iter),
                "seed": int(args.seed),
                "compact_fp16": bool(args.compact_fp16),
                "m_hnsw": int(args.m_hnsw),
                "ef_construction": int(args.ef_construction),
                "ef_search": int(args.ef_search),
            },
        )

    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, indent=2))


def run_sweep(args) -> None:
    with open(args.config_json, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    runs = config.get(args.section)
    if not isinstance(runs, list):
        raise ValueError(f"La sezione '{args.section}' non contiene una lista di run.")

    summary = []
    for run_cfg in runs:
        run_args = argparse.Namespace(
            base_h5=args.base_h5,
            query_h5=args.query_h5,
            base_dset=args.base_dset,
            query_dset=args.query_dset,
            topk=run_cfg.get("topk", args.topk),
            batch_size=run_cfg.get("batch_size", args.batch_size),
            gt_npy=args.gt_npy,
            output=None,
            output_ids_npy=None,
            output_h5=None,
            algo_name=None,
            task_name="task2",
            d=run_cfg.get("d"),
            m=run_cfg.get("m"),
            n_iter=run_cfg.get("n_iter", 7),
            seed=run_cfg.get("seed", 42),
            whiten=run_cfg.get("whiten", False),
            compact_fp16=run_cfg.get("compact_fp16", False),
            m_hnsw=run_cfg.get("m_hnsw", 32),
            ef_construction=run_cfg.get("ef_construction", 200),
            ef_search=run_cfg.get("ef_search", 64),
        )

        if args.method == "svd-rerank":
            result = capture_run(run_svd_rerank, run_args)
        elif args.method == "pca-rerank":
            result = capture_run(run_pca_rerank, run_args)
        elif args.method == "hnsw-ip":
            result = capture_run(run_hnsw_ip, run_args)
        elif args.method == "svd-hnsw-rerank":
            result = capture_run(run_svd_hnsw_rerank, run_args)
        else:
            raise ValueError(f"Metodo sweep non supportato: {args.method}")

        result["run_config"] = run_cfg
        summary.append(result)

    if args.output:
        write_json(args.output, {"method": args.method, "section": args.section, "runs": summary})
    print(json.dumps({"method": args.method, "section": args.section, "runs": summary}, indent=2))


def capture_run(fn, args):
    import io
    import contextlib

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        fn(args)
    payload = stream.getvalue().strip()
    if not payload:
        raise ValueError("La run non ha prodotto output JSON.")
    return json.loads(payload)


def summarize_metric(runs, key: str) -> Optional[dict]:
    values = [run[key] for run in runs if key in run and run[key] is not None]
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_repeat(args) -> None:
    base_kwargs = dict(
        base_h5=args.base_h5,
        query_h5=args.query_h5,
        base_dset=args.base_dset,
        query_dset=args.query_dset,
        topk=args.topk,
        batch_size=args.batch_size,
        gt_npy=args.gt_npy,
        output=None,
        output_ids_npy=None,
        output_h5=None,
        algo_name=args.algo_name,
        task_name=args.task_name,
        d=args.d,
        m=args.m,
        n_iter=args.n_iter,
        seed=args.seed,
        whiten=args.whiten,
        compact_fp16=args.compact_fp16,
        m_hnsw=args.m_hnsw,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    runs = []
    for rep in range(args.repeats):
        run_args = argparse.Namespace(**base_kwargs)
        run_args.seed = args.seed + rep
        if args.method == "svd-rerank":
            result = capture_run(run_svd_rerank, run_args)
        elif args.method == "pca-rerank":
            result = capture_run(run_pca_rerank, run_args)
        elif args.method == "hnsw-ip":
            result = capture_run(run_hnsw_ip, run_args)
        elif args.method == "svd-hnsw-rerank":
            result = capture_run(run_svd_hnsw_rerank, run_args)
        elif args.method == "exact":
            result = capture_run(run_exact, run_args)
        else:
            raise ValueError(f"Metodo repeat non supportato: {args.method}")
        result["repeat_index"] = rep
        runs.append(result)

    summary = {
        "method": args.method,
        "repeats": int(args.repeats),
        "search_time_s": summarize_metric(runs, "search_time_s"),
        "build_time_s": summarize_metric(runs, "build_time_s"),
        "qps": summarize_metric(runs, "qps"),
        "final_recall_at_k": summarize_metric(runs, "final_recall_at_k"),
        "coarse_recall_at_c": summarize_metric(runs, "coarse_recall_at_c"),
        "runs": runs,
    }

    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark scaffold per SISAP 2026 Task 2.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_cmd = sub.add_parser("inspect", help="Ispeziona struttura di un file HDF5.")
    inspect_cmd.add_argument("--h5", required=True)

    exact_cmd = sub.add_parser("exact", help="Baseline esatta con FAISS IndexFlatIP.")
    exact_cmd.add_argument("--base-h5", required=True)
    exact_cmd.add_argument("--query-h5", required=True)
    exact_cmd.add_argument("--base-dset", required=True)
    exact_cmd.add_argument("--query-dset", required=True)
    exact_cmd.add_argument("--topk", type=int, default=30)
    exact_cmd.add_argument("--batch-size", type=int, default=512)
    exact_cmd.add_argument("--output")
    exact_cmd.add_argument("--output-ids-npy")
    exact_cmd.add_argument("--output-h5")
    exact_cmd.add_argument("--algo-name")
    exact_cmd.add_argument("--task-name", default="task2")

    pca_cmd = sub.add_parser("pca-rerank", help="PCA + FlatIP sui compressi + reranking esatto.")
    pca_cmd.add_argument("--base-h5", required=True)
    pca_cmd.add_argument("--query-h5", required=True)
    pca_cmd.add_argument("--base-dset", required=True)
    pca_cmd.add_argument("--query-dset", required=True)
    pca_cmd.add_argument("--d", type=int, required=True)
    pca_cmd.add_argument("--m", type=int, required=True)
    pca_cmd.add_argument("--topk", type=int, default=30)
    pca_cmd.add_argument("--batch-size", type=int, default=512)
    pca_cmd.add_argument("--whiten", action="store_true")
    pca_cmd.add_argument("--seed", type=int, default=42)
    pca_cmd.add_argument("--gt-npy")
    pca_cmd.add_argument("--output")
    pca_cmd.add_argument("--output-ids-npy")
    pca_cmd.add_argument("--output-h5")
    pca_cmd.add_argument("--algo-name")
    pca_cmd.add_argument("--task-name", default="task2")

    svd_cmd = sub.add_parser("svd-rerank", help="TruncatedSVD + FlatIP sui compressi + reranking esatto.")
    svd_cmd.add_argument("--base-h5", required=True)
    svd_cmd.add_argument("--query-h5", required=True)
    svd_cmd.add_argument("--base-dset", required=True)
    svd_cmd.add_argument("--query-dset", required=True)
    svd_cmd.add_argument("--d", type=int, required=True)
    svd_cmd.add_argument("--m", type=int, required=True)
    svd_cmd.add_argument("--topk", type=int, default=30)
    svd_cmd.add_argument("--batch-size", type=int, default=512)
    svd_cmd.add_argument("--n-iter", type=int, default=7)
    svd_cmd.add_argument("--seed", type=int, default=42)
    svd_cmd.add_argument("--compact-fp16", action="store_true")
    svd_cmd.add_argument("--gt-npy")
    svd_cmd.add_argument("--output")
    svd_cmd.add_argument("--output-ids-npy")
    svd_cmd.add_argument("--output-h5")
    svd_cmd.add_argument("--algo-name")
    svd_cmd.add_argument("--task-name", default="task2")

    hnsw_cmd = sub.add_parser("hnsw-ip", help="Baseline HNSW con inner product.")
    hnsw_cmd.add_argument("--base-h5", required=True)
    hnsw_cmd.add_argument("--query-h5", required=True)
    hnsw_cmd.add_argument("--base-dset", required=True)
    hnsw_cmd.add_argument("--query-dset", required=True)
    hnsw_cmd.add_argument("--topk", type=int, default=30)
    hnsw_cmd.add_argument("--batch-size", type=int, default=512)
    hnsw_cmd.add_argument("--m-hnsw", type=int, default=32)
    hnsw_cmd.add_argument("--ef-construction", type=int, default=200)
    hnsw_cmd.add_argument("--ef-search", type=int, default=64)
    hnsw_cmd.add_argument("--gt-npy")
    hnsw_cmd.add_argument("--output")
    hnsw_cmd.add_argument("--output-ids-npy")
    hnsw_cmd.add_argument("--output-h5")
    hnsw_cmd.add_argument("--algo-name")
    hnsw_cmd.add_argument("--task-name", default="task2")

    svd_hnsw_cmd = sub.add_parser("svd-hnsw-rerank", help="TruncatedSVD + HNSW sui compressi + reranking esatto.")
    svd_hnsw_cmd.add_argument("--base-h5", required=True)
    svd_hnsw_cmd.add_argument("--query-h5", required=True)
    svd_hnsw_cmd.add_argument("--base-dset", required=True)
    svd_hnsw_cmd.add_argument("--query-dset", required=True)
    svd_hnsw_cmd.add_argument("--d", type=int, required=True)
    svd_hnsw_cmd.add_argument("--m", type=int, required=True)
    svd_hnsw_cmd.add_argument("--topk", type=int, default=30)
    svd_hnsw_cmd.add_argument("--batch-size", type=int, default=512)
    svd_hnsw_cmd.add_argument("--n-iter", type=int, default=7)
    svd_hnsw_cmd.add_argument("--seed", type=int, default=42)
    svd_hnsw_cmd.add_argument("--compact-fp16", action="store_true")
    svd_hnsw_cmd.add_argument("--m-hnsw", type=int, default=32)
    svd_hnsw_cmd.add_argument("--ef-construction", type=int, default=200)
    svd_hnsw_cmd.add_argument("--ef-search", type=int, default=256)
    svd_hnsw_cmd.add_argument("--gt-npy")
    svd_hnsw_cmd.add_argument("--output")
    svd_hnsw_cmd.add_argument("--output-ids-npy")
    svd_hnsw_cmd.add_argument("--output-h5")
    svd_hnsw_cmd.add_argument("--algo-name")
    svd_hnsw_cmd.add_argument("--task-name", default="task2")

    sweep_cmd = sub.add_parser("sweep", help="Esegue una lista di configurazioni da un file JSON.")
    sweep_cmd.add_argument("--method", required=True, choices=["svd-rerank", "pca-rerank", "hnsw-ip", "svd-hnsw-rerank"])
    sweep_cmd.add_argument("--config-json", required=True)
    sweep_cmd.add_argument("--section", required=True)
    sweep_cmd.add_argument("--base-h5", required=True)
    sweep_cmd.add_argument("--query-h5", required=True)
    sweep_cmd.add_argument("--base-dset", required=True)
    sweep_cmd.add_argument("--query-dset", required=True)
    sweep_cmd.add_argument("--topk", type=int, default=30)
    sweep_cmd.add_argument("--batch-size", type=int, default=512)
    sweep_cmd.add_argument("--gt-npy")
    sweep_cmd.add_argument("--output")

    repeat_cmd = sub.add_parser("repeat", help="Ripete piu volte una singola configurazione e aggrega le metriche.")
    repeat_cmd.add_argument("--method", required=True, choices=["exact", "svd-rerank", "pca-rerank", "hnsw-ip", "svd-hnsw-rerank"])
    repeat_cmd.add_argument("--repeats", type=int, default=3)
    repeat_cmd.add_argument("--base-h5", required=True)
    repeat_cmd.add_argument("--query-h5", required=True)
    repeat_cmd.add_argument("--base-dset", required=True)
    repeat_cmd.add_argument("--query-dset", required=True)
    repeat_cmd.add_argument("--topk", type=int, default=30)
    repeat_cmd.add_argument("--batch-size", type=int, default=512)
    repeat_cmd.add_argument("--gt-npy")
    repeat_cmd.add_argument("--output")
    repeat_cmd.add_argument("--algo-name")
    repeat_cmd.add_argument("--task-name", default="task2")
    repeat_cmd.add_argument("--d", type=int)
    repeat_cmd.add_argument("--m", type=int)
    repeat_cmd.add_argument("--n-iter", type=int, default=7)
    repeat_cmd.add_argument("--seed", type=int, default=42)
    repeat_cmd.add_argument("--whiten", action="store_true")
    repeat_cmd.add_argument("--compact-fp16", action="store_true")
    repeat_cmd.add_argument("--m-hnsw", type=int, default=32)
    repeat_cmd.add_argument("--ef-construction", type=int, default=200)
    repeat_cmd.add_argument("--ef-search", type=int, default=64)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "inspect":
        inspect_h5(args.h5)
    elif args.cmd == "exact":
        run_exact(args)
    elif args.cmd == "pca-rerank":
        run_pca_rerank(args)
    elif args.cmd == "svd-rerank":
        run_svd_rerank(args)
    elif args.cmd == "hnsw-ip":
        run_hnsw_ip(args)
    elif args.cmd == "svd-hnsw-rerank":
        run_svd_hnsw_rerank(args)
    elif args.cmd == "sweep":
        run_sweep(args)
    elif args.cmd == "repeat":
        run_repeat(args)
    else:  # pragma: no cover
        raise ValueError(f"Comando sconosciuto: {args.cmd}")


if __name__ == "__main__":
    main()
