import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import umap
import hdbscan


DEFAULT_TAUS = [0.01, 0.05, 0.1, 0.5]


def parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_params(dataset: str | None, config_path: str | None):
    """
    Loads UMAP+HDBSCAN params.
    Priority:
      1) --config
      2) --dataset (from configs/paper_params.json)
      3) auto mode (None returned -> auto heuristics)
    """
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # allow either top-level {"umap":..., "hdbscan":...} OR {"adult":{...}}
        if "umap" in cfg and "hdbscan" in cfg:
            return cfg["umap"], cfg["hdbscan"], "custom_config"
        if dataset and dataset in cfg:
            return cfg[dataset]["umap"], cfg[dataset]["hdbscan"], "custom_config"
        raise ValueError("Config file loaded but dataset key not found and no top-level umap/hdbscan present.")

    if dataset:
        paper_path = Path("configs") / "paper_params.json"
        if not paper_path.exists():
            raise FileNotFoundError("configs/paper_params.json not found. Please create it first.")
        with open(paper_path, "r", encoding="utf-8") as f:
            paper = json.load(f)
        if dataset not in paper:
            raise ValueError(f"Unknown dataset '{dataset}'. Use one of: {list(paper.keys())}, or omit --dataset for auto mode.")
        return paper[dataset]["umap"], paper[dataset]["hdbscan"], "paper_defaults"

    return None, None, "auto"


def build_mixed_encoded(real_df: pd.DataFrame, syn_df: pd.DataFrame):
    """
    Mixed-type encoding:
    - numeric: MinMax scaling (fit on real)
    - categorical: one-hot (based on concatenated real+syn so columns match)
    Returns:
      X_real (np.ndarray), X_syn (np.ndarray), feature_names (list[str])
    """
    # Identify numeric columns
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    # Treat everything else as categorical
    cat_cols = [c for c in real_df.columns if c not in numeric_cols]

    # Scale numeric using real fit
    scaler = MinMaxScaler()
    real_num = real_df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=real_df.index)
    syn_num = syn_df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=syn_df.index)

    if numeric_cols:
        scaler.fit(real_num.values)
        real_num_scaled = pd.DataFrame(scaler.transform(real_num.values), columns=numeric_cols, index=real_df.index)
        syn_num_scaled = pd.DataFrame(scaler.transform(syn_num.values), columns=numeric_cols, index=syn_df.index)
    else:
        real_num_scaled = real_num
        syn_num_scaled = syn_num

    # One-hot categoricals using combined (ensures same dummy columns)
    if cat_cols:
        real_cat = real_df[cat_cols].astype(str).copy()
        syn_cat = syn_df[cat_cols].astype(str).copy()
        combined = pd.concat([real_cat, syn_cat], axis=0, ignore_index=True)
        combined_dummies = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
        real_cat_oh = combined_dummies.iloc[: len(real_df), :].set_index(real_df.index)
        syn_cat_oh = combined_dummies.iloc[len(real_df) :, :].set_index(syn_df.index)
    else:
        real_cat_oh = pd.DataFrame(index=real_df.index)
        syn_cat_oh = pd.DataFrame(index=syn_df.index)

    # Combine
    real_enc = pd.concat([real_num_scaled, real_cat_oh], axis=1)
    syn_enc = pd.concat([syn_num_scaled, syn_cat_oh], axis=1)

    # Align columns (safety)
    real_enc, syn_enc = real_enc.align(syn_enc, join="outer", axis=1, fill_value=0.0)

    feature_names = real_enc.columns.tolist()
    return real_enc.values.astype(np.float32), syn_enc.values.astype(np.float32), feature_names


def auto_params(n_syn: int, dim: int):
    """
    Simple deterministic auto settings.
    Goal: works out-of-the-box, not perfect tuning.
    """
    n_neighbors = int(np.clip(np.sqrt(max(n_syn, 1)), 10, 50))
    n_components = int(np.clip(np.log2(dim + 1), 5, 15))
    umap_params = {
        "n_neighbors": n_neighbors,
        "min_dist": 0.1,
        "n_components": n_components,
        "random_state": 42,
    }
    min_cluster_size = max(20, int(0.01 * n_syn))
    min_samples = max(5, int(0.002 * n_syn))
    hdb_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }
    return umap_params, hdb_params


def compute_medoids_from_embedding(embedding: np.ndarray, labels: np.ndarray):
    """
    Compute one medoid per cluster (excluding noise=-1), in embedding space.
    Medoid = point minimizing sum of distances to others in cluster.
    Returns: list of indices (into synthetic rows) corresponding to medoids.
    """
    medoid_indices = []
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        # For small/medium clusters, exact medoid:
        # compute pairwise distances in embedding space
        X = embedding[idx]
        # squared euclidean via dot trick (stable enough for embedding dims)
        # dist^2(i,j) = ||xi||^2 + ||xj||^2 - 2 xi·xj
        norms = np.sum(X * X, axis=1, keepdims=True)
        d2 = norms + norms.T - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)
        # sum distances (use sqrt for true euclidean; monotone but keep sqrt for correctness)
        d = np.sqrt(d2)
        sums = np.sum(d, axis=1)
        medoid_local = int(np.argmin(sums))
        medoid_indices.append(int(idx[medoid_local]))

    return medoid_indices


def summarize_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {"M": 0}
    return {
        "M": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def main():
    parser = argparse.ArgumentParser(description="Medoid-based evaluator for real vs synthetic tabular data.")
    parser.add_argument("--real", required=True, help="Path to real.csv")
    parser.add_argument("--syn", required=True, help="Path to synthetic.csv")
    parser.add_argument("--out", default="metrics.json", help="Output JSON path (default: metrics.json)")
    parser.add_argument("--dataset", default=None, help="One of: adult, bank, telco, drug (loads paper params).")
    parser.add_argument("--config", default=None, help="Optional JSON config with umap/hdbscan params.")
    parser.add_argument("--taus", default="0.01,0.05,0.1,0.5", help="Comma-separated tau thresholds.")
    parser.add_argument("--ignore-cols", default="", help="Comma-separated columns to ignore (e.g., id,index).")
    args = parser.parse_args()

    taus = parse_float_list(args.taus) if args.taus else DEFAULT_TAUS
    ignore_cols = set(parse_list(args.ignore_cols)) if args.ignore_cols else set()

    real_df = pd.read_csv(args.real)
    syn_df = pd.read_csv(args.syn)

    # Drop ignored columns if present
    for c in list(ignore_cols):
        if c in real_df.columns:
            real_df = real_df.drop(columns=[c])
        if c in syn_df.columns:
            syn_df = syn_df.drop(columns=[c])

    # Validate column sets
    if set(real_df.columns) != set(syn_df.columns):
        missing_in_syn = sorted(list(set(real_df.columns) - set(syn_df.columns)))
        missing_in_real = sorted(list(set(syn_df.columns) - set(real_df.columns)))
        raise ValueError(
            "real.csv and synthetic.csv must have the same columns.\n"
            f"Missing in synthetic: {missing_in_syn}\n"
            f"Missing in real: {missing_in_real}"
        )

    # Align order
    syn_df = syn_df[real_df.columns.tolist()]

    # Drop rows with missing values
    n_real0, n_syn0 = len(real_df), len(syn_df)
    real_df = real_df.dropna()
    syn_df = syn_df.dropna()
    dropped_real = n_real0 - len(real_df)
    dropped_syn = n_syn0 - len(syn_df)

    # Encode mixed types
    X_real, X_syn, feature_names = build_mixed_encoded(real_df, syn_df)

    # Params
    umap_params, hdb_params, mode = load_params(args.dataset, args.config)
    if umap_params is None or hdb_params is None:
        umap_params, hdb_params = auto_params(n_syn=len(X_syn), dim=X_syn.shape[1])

    # UMAP on synthetic only
    umap_model = umap.UMAP(
        n_neighbors=int(umap_params["n_neighbors"]),
        min_dist=float(umap_params["min_dist"]),
        n_components=int(umap_params["n_components"]),
        random_state=int(umap_params.get("random_state", 42)),
    )
    embedding = umap_model.fit_transform(X_syn)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hdb_params["min_cluster_size"]),
        min_samples=int(hdb_params["min_samples"]),
        prediction_data=False,
    )
    labels = clusterer.fit_predict(embedding)

    # Medoids (indices into synthetic)
    medoid_idx = compute_medoids_from_embedding(embedding, labels)
    X_medoids = X_syn[medoid_idx] if len(medoid_idx) > 0 else np.zeros((0, X_syn.shape[1]), dtype=np.float32)

    # Nearest real for each medoid -> dmin
    if len(X_medoids) > 0:
        nn_real = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nn_real.fit(X_real)
        dmin, _ = nn_real.kneighbors(X_medoids, n_neighbors=1, return_distance=True)
        dmin = dmin.reshape(-1)
    else:
        dmin = np.array([], dtype=float)

    # Coverage: for each real row, distance to nearest medoid
    if len(X_medoids) > 0:
        nn_med = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nn_med.fit(X_medoids)
        d_real_to_med, _ = nn_med.kneighbors(X_real, n_neighbors=1, return_distance=True)
        d_real_to_med = d_real_to_med.reshape(-1)
    else:
        d_real_to_med = np.full((len(X_real),), np.inf, dtype=float)

    # Metrics
    asr = {str(t): float(np.mean(dmin <= t)) if dmin.size > 0 else 0.0 for t in taus}
    coverage = {str(t): float(np.mean(d_real_to_med <= t)) if d_real_to_med.size > 0 else 0.0 for t in taus}

    out = {
        "mode": mode,
        "dataset": args.dataset if args.dataset else None,
        "taus": taus,
        "n_real": int(len(real_df)),
        "n_syn": int(len(syn_df)),
        "n_features_encoded": int(X_syn.shape[1]),
        "n_medoids": int(len(medoid_idx)),
        "asr": asr,
        "coverage": coverage,
        "dmin_stats": summarize_stats(dmin),
        "dropped_rows": {"real": int(dropped_real), "syn": int(dropped_syn)},
        "params_used": {"umap": umap_params, "hdbscan": hdb_params},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("✅ Done. Wrote:", str(out_path))
    print("  mode:", out["mode"])
    print("  n_real / n_syn:", out["n_real"], "/", out["n_syn"])
    print("  n_medoids:", out["n_medoids"])
    print("  ASR:", out["asr"])
    print("  Coverage:", out["coverage"])
    print("  dmin_stats:", out["dmin_stats"])


if __name__ == "__main__":
    main()
