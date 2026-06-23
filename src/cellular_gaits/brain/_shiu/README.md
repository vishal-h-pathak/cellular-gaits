# Vendored: Shiu *Drosophila* brain model (v783)

This directory is a **verbatim vendor** of the connectome leaky-integrate-and-fire
(LIF) model published by Philip Shiu & Nico Spiller.

- **Upstream:** https://github.com/philshiu/Drosophila_brain_model (branch `main`)
- **License:** MIT — see [`LICENSE`](LICENSE) (Copyright (c) 2023 Philip Shiu and Nico Spiller)
- **Paper:** Shiu, Spiller, et al., *A leaky integrate-and-fire computational model based
  on the connectome of the entire adult Drosophila brain reveals insights into
  sensorimotor processing*, Nature 2024 (biorxiv `2023.05.02.539144`).
- **Connectome:** FlyWire **v783** (Dorkenwald et al. 2024); neurotransmitter signs
  predicted per Eckstein et al. 2024.

## What's here

| File | Provenance | Notes |
|---|---|---|
| `model.py` | upstream, **unmodified** | Builds the Brian2 network from the connectome; activate / silence / run. |
| `utils.py` | upstream, **unmodified** | Spike-train → firing-rate helpers. |
| `data/Completeness_783.csv` | upstream (~3.2 MB) | The neuron list (FlyWire IDs). **Committed.** |
| `data/Connectivity_783.parquet` | upstream (~97 MB) | The synapse table (pre→post, weight, NT sign). **Gitignored** — run `fetch_data.sh`. |

`model.py` / `utils.py` are kept byte-for-byte identical to upstream so the published
LIF dynamics are unchanged. The project's clean wrapper API lives one level up in
`cellular_gaits.brain.brain_model` and only *calls* these functions.

## Getting the heavy data

The 97 MB connectivity parquet is not in git. After cloning:

```bash
bash src/cellular_gaits/brain/_shiu/fetch_data.sh
```

This pulls `Connectivity_783.parquet` (and, if missing, the CSV) from the public
upstream repo into `data/`.
