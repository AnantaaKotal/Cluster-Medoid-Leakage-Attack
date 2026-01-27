# Cluster–Medoid Leakage Attack (CMLA) — Minimal Evaluator

This repository is the official evaluation code for the paper:

**When Privacy Isn’t Synthetic: Hidden Data Leakage in Generative AI Models**  
S M Mustaquim, Anantaa Kotal, Paul H. Yi  
arXiv:2512.06062 — https://arxiv.org/abs/2512.06062

It implements the paper’s **Cluster–Medoid Leakage Attack (CMLA)** idea: cluster synthetic samples (UMAP + HDBSCAN), extract **cluster medoids**, and audit **nearest-neighbor distances** to real data.

 **Input:** `real.csv` and `synthetic.csv` (same columns)  
 **Output:** exactly 3 evaluation signals
- **ASR(τ)**: fraction of synthetic **medoids** within distance τ of the nearest real record
- **Coverage(τ)**: fraction of real records within distance τ of at least one synthetic medoid
- **dmin summary stats**: min/mean/median/max/p10/p90 of medoid→real nearest-neighbor distances

Supports **mixed numeric + categorical** tabular data.

---

## How to run

### Paper datasets (frozen parameters)

```bash
python evaluate.py --dataset adult --real real.csv --syn synthetic.csv
```
Supported datasets: `adult`, `bank`, `telco`, `drug`

#### Run the included Adult example

```bash
python evaluate.py --dataset adult --real examples/adult_real.csv --syn examples/adult_synthetic_GReat.csv
```

---

### Any new dataset (auto mode)

```bash
python evaluate.py --real real.csv --syn synthetic.csv
```
## Input requirements

- `real.csv` and `synthetic.csv` must have the **same column names** and meaning.
- Mixed types are supported:
  - numeric columns: int/float
  - categorical columns: string/bool/object

Ignore identifier columns (optional):

```bash
python evaluate.py --real real.csv --syn synthetic.csv --ignore-cols id,index
```
Missing values: rows with missing values are dropped and the drop counts are reported.

---

## Output

The script prints a short summary and writes `metrics.json` containing:
- ASR(τ) values
- Coverage(τ) values
- dmin statistics (min/mean/median/max/p10/p90)
- number of medoids
- parameters used (paper defaults or auto mode)

---

## Citation

If you use this code, please cite:

```bibtex
@article{mustaqim2025privacy,
  title   = {When Privacy Isn't Synthetic: Hidden Data Leakage in Generative AI Models},
  author  = {Mustaqim, S. M. and Kotal, Anantaa and Yi, Paul H.},
  journal = {arXiv preprint arXiv:2512.06062},
  year    = {2025}
}
```
