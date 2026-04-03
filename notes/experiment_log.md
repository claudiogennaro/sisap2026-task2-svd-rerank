# Experiment Log

## Summary so far

- Obiettivo iniziale: partecipare al `Task 2` della SISAP 2026, capire requisiti, dataset, formato output e vincoli Docker/risorse.
- Dataset usato per lo sviluppo: `llama-128-ip.hdf5`, con `train` `(256921, 128)` e `test` `(1000, 128)`.
- Baseline esatta con `FAISS IndexFlatIP`: circa `1.03s` su `1000` query, usata anche per generare il ground truth `runs/exact_ids.npy`.
- Prima idea testata: `PCA + FlatIP sui compressi + rerank`.
- Esito della PCA: fallimento netto per MIPS, coarse recall quasi nulla; conclusione che la PCA centrata non preserva il ranking per inner product.
- Baseline `HNSW` puro su vettori originali: molto veloce ma recall troppo bassa anche con parametri più spinti; scartata come soluzione principale.
- Seconda idea testata: `TruncatedSVD + FlatIP sui compressi + rerank`.
- Sweep iniziale su `D` con `M=8`: miglioramento regolare della recall al crescere di `D`; questa famiglia si è rivelata promettente.
- Sweep successivo su `M`: identificati punti sopra soglia `0.8`, in particolare `D=80, M=8` e `D=64, M=12/16`.
- Correzione della metrica `coarse_recall_at_c`: inizialmente misurava solo i primi `30` candidati, poi corretta per considerare tutti i `k*M` candidati.
- Raffinamento locale attorno al best point: provati `D=72,76,78,80` e `M=6,7,8,10`.
- Risultato importante: `D=78, M=8` era veloce ma troppo fragile rispetto al seed, con alcune run sotto `0.8`.
- Test di stabilita con `repeat`: `D=76, M=10` si e rivelata la configurazione piu robusta, con recall circa `0.831` stabile su piu seed.
- Aggiunta esportazione HDF5 challenge-style: `knns`, `dists`, attributi root `algo`, `task`, `buildtime`, `querytime`, `params`.
- Creazione del runner finale `src/run_task2.py` per produrre direttamente `results/task2/*.h5`.
- Creazione `Dockerfile` e verifica end-to-end del container.
- Primo test Docker: problema iniziale di mount path; risolto allineando i volumi a `/app/data` e `/app/results` come da challenge.
- Verifica dell’output finale HDF5: file valido con dataset e attributi corretti.
- Pulizia del progetto per la submission: `.gitignore`, GitHub Actions `Smoke Test`, README orientato alla submission, repo GitHub pubblico.
- Registrazione effettuata: issue di pre-registration aperta per il team `Viento Norte` sul `Task 2`.
- Nuova idea testata: `SVD + HNSW sui compressi + rerank`.
- Esito della variante `SVD + HNSW`: recall crollata a circa `0.08 - 0.13`; scartata.
- Ottimizzazione mirata finale: test su `batch_size` e `compact-fp16`.
- Esito dei test mirati:
  - `batch_size=1024` migliora molto il throughput mantenendo la stessa recall.
  - `compact-fp16` non porta benefici reali.
- Configurazione finale ufficiale del progetto:
  - `svd-rerank`
  - `d=76`
  - `m=10`
  - `topk=30`
  - `batch_size=1024`

## Update policy

- Aggiungere qui solo esperimenti che portano a una decisione tecnica o eliminano una linea di ricerca.
- Evitare di riportare benchmark ridondanti o rumorosi senza impatto sulle scelte.
