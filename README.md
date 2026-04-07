# SISAP 2026 Task 2 Submission Scaffold

Repository for `Task 2` of the SISAP 2026 Indexing Challenge.

The current solution uses:
- `TruncatedSVD` for linear dimensionality reduction
- `FlatIP` search in the compressed space
- exact reranking on the original vectors

The most robust configuration found so far is:
- `d=76`
- `m=10`
- `topk=30`
- `batch_size=1024`

## Task 2 hyperparameter sets

The current candidate submission configurations are collected in:
- `experiments/task2_submission.json`

Current candidate sets:
- `svd-rerank-primary`: `d=76`, `m=10`, `topk=30`, `batch_size=1024`, `n_iter=7`
- `svd-rerank-alt-80x8`: `d=80`, `m=8`, `topk=30`, `batch_size=1024`, `n_iter=7`
- `svd-rerank-alt-76x10-bs512`: `d=76`, `m=10`, `topk=30`, `batch_size=512`, `n_iter=7`

The number of sets is intentionally small at the moment, but the format is already ready to scale up to 15 configurations.

## Repository contents

- `src/task2_bench.py`: benchmark runner for exact, HNSW, PCA/SVD rerank methods
- `src/run_task2.py`: final runner for `Task 2`
- `experiments/task2_initial.json`: exploratory experiment sets
- `experiments/task2_submission.json`: current candidate submission sets
- `Dockerfile`: reproducible container image

## Environment

Local installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Input layout

The container follows the layout expected by the challenge:
- input dataset in `/app/data`
- output in `/app/results`

For the development file used so far:
- HDF5 file: `llama-128-ip.hdf5`
- base dataset: `train`
- query dataset: `test`

## Run locally

```bash
python src/run_task2.py \
  --input-h5 data/llama-128-ip.hdf5 \
  --output-h5 results/task2/svd_d76_m10.h5
```

## Build Docker image

```bash
docker build -t sisap-task2 .
```

## Run in Docker

```bash
docker run --rm \
  --cpus=8 \
  --memory=24g \
  --memory-swap=24g \
  --memory-swappiness=0 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  sisap-task2 \
  --input-h5 /app/data/llama-128-ip.hdf5 \
  --output-h5 /app/results/task2/svd_d76_m10.h5
```

## Output format

The HDF5 output written by the runner contains:
- `knns`: shape `(n_queries, 30)`, `1-based` identifiers
- `dists`: shape `(n_queries, 30)`

Root attributes:
- `algo`
- `task`
- `buildtime`
- `querytime`
- `params`

## Benchmark utilities

Inspect an HDF5 file:

```bash
python src/task2_bench.py inspect --h5 data/llama-128-ip.hdf5
```

Exact baseline:

```bash
python src/task2_bench.py exact \
  --base-h5 data/llama-128-ip.hdf5 \
  --query-h5 data/llama-128-ip.hdf5 \
  --base-dset train \
  --query-dset test \
  --topk 30 \
  --batch-size 512 \
  --output runs/exact.json \
  --output-ids-npy runs/exact_ids.npy
```

SVD rerank:

```bash
python src/task2_bench.py svd-rerank \
  --base-h5 data/llama-128-ip.hdf5 \
  --query-h5 data/llama-128-ip.hdf5 \
  --base-dset train \
  --query-dset test \
  --d 76 \
  --m 10 \
  --topk 30 \
  --batch-size 1024 \
  --gt-npy runs/exact_ids.npy \
  --output runs/svd_d76_m10.json
```

To test compressed-vector storage in `float16`, add:

```bash
  --compact-fp16
```

SVD + HNSW rerank:

```bash
python src/task2_bench.py svd-hnsw-rerank \
  --base-h5 data/llama-128-ip.hdf5 \
  --query-h5 data/llama-128-ip.hdf5 \
  --base-dset train \
  --query-dset test \
  --d 76 \
  --m 10 \
  --m-hnsw 32 \
  --ef-construction 200 \
  --ef-search 256 \
  --topk 30 \
  --batch-size 1024 \
  --gt-npy runs/exact_ids.npy \
  --output runs/svd_hnsw_d76_m10.json
```

Repeat:

```bash
python src/task2_bench.py repeat \
  --method svd-rerank \
  --repeats 3 \
  --base-h5 data/llama-128-ip.hdf5 \
  --query-h5 data/llama-128-ip.hdf5 \
  --base-dset train \
  --query-dset test \
  --d 76 \
  --m 10 \
  --topk 30 \
  --batch-size 1024 \
  --gt-npy runs/exact_ids.npy \
  --output runs/svd_d76_m10_repeat.json
```

Targeted experiments on `batch_size` and `compact-fp16`:

```bash
python scripts/run_targeted_svd_experiments.py
```

## Notes

- The `Dockerfile` fixes the main threading-related environment variables to `8`.
- The benchmark runner and the final runner share the same execution code path.
- The repository does not include datasets, results, or local environments.
