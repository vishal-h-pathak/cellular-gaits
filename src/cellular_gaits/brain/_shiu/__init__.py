"""Vendored, unmodified Shiu connectome LIF model.

Source: https://github.com/philshiu/Drosophila_brain_model (MIT, see LICENSE).
Paper:  Shiu, Spiller et al. 2024, Nature (and biorxiv 2023.05.02.539144).

`model.py` and `utils.py` are copied verbatim from upstream `main`. Do NOT edit
them — the project's clean API lives one level up in
`cellular_gaits.brain.brain_model` and only *calls* these functions, so the
published leaky-integrate-and-fire dynamics stay exactly as their authors wrote
them. See `README.md` here for provenance and the data-fetch instructions.
"""
