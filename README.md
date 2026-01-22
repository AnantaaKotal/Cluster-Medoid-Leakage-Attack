# cluster-medoid-leakage-attack

A minimal evaluator for memorization / leakage risk in tabular synthetic data.

✅ **You provide**
1. `real.csv` (your real dataset)
2. `synthetic.csv` (your synthetic dataset)

✅ **You get 3 metrics**
- **ASR(τ)**: fraction of synthetic **medoids** within distance τ of the nearest real record
- **Coverage(τ)**: fraction of real records within distance τ of at least one synthetic medoid
- **dmin summary stats**: min/mean/median/max/p10/p90 of medoid-to-real nearest-neighbor distances

Supports **mixed numeric + categorical** tabular data.

---

## How to run

### Paper datasets (frozen parameters)

```bash
python evaluate.py --dataset adult --real real.csv --syn synthetic.csv
```
Supported datasets: `adult`, `bank`, `telco`, `drug`

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

If you use this evaluator in your work, please cite the corresponding paper this repository accompanies.
