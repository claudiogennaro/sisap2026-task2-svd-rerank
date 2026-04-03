# SISAP 2026 Task 2 Submission Scaffold

Repository per il `Task 2` della SISAP 2026 Indexing Challenge.

La soluzione corrente usa:
- `TruncatedSVD` per riduzione lineare
- ricerca `FlatIP` nello spazio compresso
- reranking esatto sui candidati nello spazio originale

La configurazione piu robusta trovata finora e:
- `d=76`
- `m=10`
- `topk=30`
- `batch_size=1024`

## Repository contents

- `src/task2_bench.py`: benchmark runner per exact, HNSW, PCA/SVD rerank
- `src/run_task2.py`: runner finale per il `Task 2`
- `experiments/task2_initial.json`: set iniziali di esperimenti
- `Dockerfile`: immagine container per esecuzione riproducibile

## Environment

Installazione locale:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Input layout

Il container segue il layout atteso dalla challenge:
- input dataset in `/app/data`
- output in `/app/results`

Per il file di sviluppo usato finora:
- file HDF5: `llama-128-ip.hdf5`
- dataset base: `train`
- dataset query: `test`

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

L'output HDF5 scritto dal runner contiene:
- `knns`: shape `(n_queries, 30)`, ID `1-based`
- `dists`: shape `(n_queries, 30)`

Attributi root:
- `algo`
- `task`
- `buildtime`
- `querytime`
- `params`

## Benchmark utilities

Ispezione HDF5:

```bash
python src/task2_bench.py inspect --h5 data/llama-128-ip.hdf5
```

Baseline esatta:

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
  --batch-size 512 \
  --gt-npy runs/exact_ids.npy \
  --output runs/svd_d76_m10.json
```

Per testare la memorizzazione dei vettori compressi in `float16`, aggiungi:

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
  --batch-size 512 \
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
  --batch-size 512 \
  --gt-npy runs/exact_ids.npy \
  --output runs/svd_d76_m10_repeat.json
```

Esperimenti mirati su `batch_size` e `compact-fp16`:

```bash
python scripts/run_targeted_svd_experiments.py
```

## Notes

- Il `Dockerfile` fissa le principali variabili di threading a `8`.
- Il benchmark e il runner finale condividono lo stesso codice di esecuzione.
- Il repository non include dataset, risultati o ambienti locali.
