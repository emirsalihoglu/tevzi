# Tevzi

Adaptive multimodal resource scheduling for GPU clusters.

Tevzi predicts each job's **duration** and **actual resource usage** from two modalities —
static job features (tabular) and cluster dynamics (time series) — fused with
state-dependent (adaptive) weights. The predictions drive a simple, transparent scheduler,
and their contribution to **resource efficiency** and **job completion time** is measured
in a discrete-event simulation built on the Alibaba `cluster-trace-gpu-v2020`.

Project plan (Turkish): [docs/PROJE_PLANI.md](docs/PROJE_PLANI.md)

## Repository layout
```
data/                raw / interim / processed (not tracked; data lives on Google Drive)
notebooks/           exploration and Colab notebooks
src/tevzi/
  data/              loading, cleaning, temporal split
  features/          static job features (M1), cluster time series (M2)
  models/            GBM baseline, MLP, sequence encoders, fixed/adaptive fusion
  scheduler/         ordering + best-fit placement + safety margin
  simulator/         discrete-event cluster simulator, penalty model
  evaluation/        prediction and scheduling metrics
experiments/configs/ experiment configs (YAML)
results/             models, tables, figures
reports/             report drafts
docs/                plan and notes
```

## Data
Alibaba Cluster Trace — `cluster-trace-gpu-v2020`
(Weng et al., *MLaaS in the Wild*, NSDI 2022).