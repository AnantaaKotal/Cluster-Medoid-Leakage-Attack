# cluster-medoid-leakage-attack

A minimal evaluator for memorization / leakage risk in tabular synthetic data.

✅ You only provide:
1) `real.csv` (your real dataset)
2) `synthetic.csv` (your synthetic dataset)

…and this tool outputs **three metrics**:
- **ASR(τ)**: fraction of synthetic **medoids** within distance τ of the nearest real record
- **Coverage(τ)**: fraction of real records within distance τ of at least one synthetic medoid
- **dmin summary stats**: min/mean/median/max/p10/p90 of medoid-to-real nearest-neighbor distances

This tool supports **mixed numeric + categorical** tabular data.

---

## How to run

### Paper datasets (frozen parameters)
Use the same UMAP + HDBSCAN parameters used in the paper experiments:

```bash
python evaluate.py --dataset adult --real real.csv --syn synthetic.csv
