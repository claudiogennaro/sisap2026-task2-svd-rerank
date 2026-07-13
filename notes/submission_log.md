# Submission Log

## TIRA code submission

- Date: 2026-06-11
- Task: `sisap-2026`
- Team: `viento-norte`
- Submission/software name: `flashed-buyout`
- Git repository: `git@github.com:claudiogennaro/sisap2026-task2-svd-rerank.git`
- Branch: `main`
- Commit: `7414207838fa3300b49026439eb548257ee382b7`

## Registered command

```bash
python src/run_task2.py --input "$inputDataset/*.h5" --task-description "$inputDataset/config.json" --output "$outputDir"
```

## Notes

- The TIRA dry run passed on `task-2-spot-check-20260528-training`.
- The real code submission was uploaded successfully to TIRA.
- The submission is visible in the TIRA UI under `sisap-2026` for team `viento-norte`.

## TIRA code resubmission

- Date: 2026-07-13
- Task: `sisap-2026`
- Team: `viento-norte`
- Submission/software name: `generous-packet`
- Git repository: `git@github.com:claudiogennaro/sisap2026-task2-svd-rerank.git`
- Branch: `main`
- Commit: `57156619e269456ce5430a2f378126faadbe6e6b`

## Registered command (resubmission)

```bash
python src/run_task2.py --input "$inputDataset" --task-description "$inputDataset/config.json" --output "$outputDir"
```

## Resubmission Notes

- The original submission failed Task 2 evaluation because the runner did not robustly resolve the dataset layout used by the current TIRA setup.
- The runner was updated to:
  - support nested HDF5 dataset paths such as `test/queries`
  - support directory-style TIRA inputs
  - resolve datasets from `config.json` metadata
  - resolve mirrored resource manifests when the HDF5 file is referenced but not directly present in the input directory
  - write the `dataset` HDF5 root attribute expected by the public evaluation script
- A dry run passed successfully on `task-2-llama-dev-20260630-training`.
- The corrected code submission was uploaded successfully to TIRA as `generous-packet`.
