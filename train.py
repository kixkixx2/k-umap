"""Train the UMAP + K-Means clustering pipeline and persist artifacts."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import umap.umap_ as umap
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data") / "final_real.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory where trained objects and metadata will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible UMAP/K-Means training.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run while tuning hyperparameters.",
    )
    parser.add_argument(
        "--high-card-threshold",
        type=int,
        default=20,
        help="Drop categorical columns with more unique values than this threshold.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataset: shape={df.shape}")
    return df


def drop_high_cardinality(df: pd.DataFrame, threshold: int) -> Tuple[pd.DataFrame, List[str]]:
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique(dropna=False) > threshold]
    if high_card_cols:
        print(f"Dropping {len(high_card_cols)} high-cardinality columns (> {threshold} unique)")
        df = df.drop(columns=high_card_cols)
    else:
        print("No high-cardinality categorical columns detected.")
    return df, high_card_cols


def frequency_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    freq_maps: Dict[str, Dict] = {}
    if not cat_cols:
        print("No categorical columns left to encode.")
        return df, freq_maps

    print(f"Applying frequency encoding to {len(cat_cols)} categorical columns")
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        freq_map = (counts / len(df)).to_dict()
        freq_maps[col] = freq_map
        df[col] = df[col].map(freq_map).fillna(0)
        print(f"  Encoded {col} ({len(freq_map)} categories)")
    return df, freq_maps


def clean_data(df: pd.DataFrame, n_neighbors: int = 5) -> Tuple[pd.DataFrame, np.ndarray, Dict, KNNImputer, StandardScaler, List[str]]:
    start = time.time()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    imputer = KNNImputer(n_neighbors=n_neighbors) if num_cols else None
    scaler = StandardScaler() if num_cols else None

    data = df.copy()
    if num_cols and imputer is not None:
        data[num_cols] = imputer.fit_transform(data[num_cols])
    scaled = scaler.fit_transform(data[num_cols]) if num_cols and scaler is not None else np.empty((len(data), 0))

    report = {
        "original_shape": tuple(df.shape),
        "final_shape": tuple(data.shape),
        "imputed_values": int(np.isnan(df[num_cols]).sum().sum() if num_cols else 0),
        "retention_pct": 100.0,
        "duration_sec": time.time() - start,
    }
    print(
        f"Cleaned data: original={report['original_shape']} final={report['final_shape']} duration={report['duration_sec']:.2f}s"
    )
    return data.reset_index(drop=True), scaled, report, imputer, scaler, num_cols


def run_optuna(X_scaled: np.ndarray, random_state: int, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
            "min_dist": trial.suggest_float("min_dist", 0.0, 0.2),
            "n_components": trial.suggest_categorical("n_components", [2, 3, 5, 8, 10]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "cosine", "manhattan"]),
            "n_clusters": trial.suggest_int("n_clusters", 2, 12),
        }
        try:
            reducer = umap.UMAP(
                n_neighbors=params["n_neighbors"],
                min_dist=params["min_dist"],
                n_components=params["n_components"],
                metric=params["metric"],
                random_state=random_state,
                verbose=False,
            )
            embedding = reducer.fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=params["n_clusters"], random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embedding)
            return silhouette_score(embedding, labels)
        except Exception:
            return -1.0

    print(f"Running Optuna hyperparameter search for {n_trials} trials…")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    duration = time.time() - start
    print(
        f"Optuna complete: best_score={study.best_value:.4f} duration={duration:.1f}s trials={len(study.trials)}"
    )
    return study


def train_final_models(
    X_scaled: np.ndarray, best_params: Dict, random_state: int
) -> Tuple[np.ndarray, np.ndarray, umap.UMAP, KMeans, float]:
    start = time.time()
    reducer = umap.UMAP(
        n_neighbors=best_params["n_neighbors"],
        min_dist=best_params["min_dist"],
        n_components=best_params["n_components"],
        metric=best_params["metric"],
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=best_params["n_clusters"], random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embedding)
    duration = time.time() - start
    print(f"Final training runtime: {duration:.2f}s")
    return embedding, labels, reducer, kmeans, duration


def compute_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    kmeans: KMeans,
    runtime: float,
) -> Dict:
    silhouette = silhouette_score(embedding, labels)
    db_index = davies_bouldin_score(embedding, labels)
    cluster_sizes = {int(idx): int(count) for idx, count in enumerate(np.bincount(labels))}

    centers = kmeans.cluster_centers_
    centroid_distances = cdist(centers, centers)
    nonzero = centroid_distances[np.nonzero(centroid_distances)]
    min_separation = float(nonzero.min()) if nonzero.size else 0.0
    mean_distance = float(nonzero.mean()) if nonzero.size else 0.0

    metrics = {
        "silhouette": float(silhouette),
        "davies_bouldin": float(db_index),
        "cluster_sizes": cluster_sizes,
        "runtime_sec": float(runtime),
        "time_per_sample_ms": float((runtime / len(embedding)) * 1000),
        "throughput_samples_per_sec": float(len(embedding) / runtime) if runtime else 0.0,
        "min_centroid_separation": min_separation,
        "mean_centroid_distance": mean_distance,
    }
    print(
        f"Metrics → silhouette={metrics['silhouette']:.4f} | DB={metrics['davies_bouldin']:.4f} | runtime={metrics['runtime_sec']:.2f}s"
    )
    return metrics


def save_artifacts(
    artifact_dir: Path,
    reducer: umap.UMAP,
    kmeans: KMeans,
    imputer: KNNImputer,
    scaler: StandardScaler,
    freq_maps: Dict,
    metadata: Dict,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, artifact_dir / "umap_reducer.joblib")
    joblib.dump(kmeans, artifact_dir / "kmeans_model.joblib")
    joblib.dump(imputer, artifact_dir / "knn_imputer.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(freq_maps, artifact_dir / "frequency_encoding_maps.joblib")

    with open(artifact_dir / "training_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)
    print(f"Artifacts saved to {artifact_dir}")


def _json_default(obj):  # pragma: no cover - helper for JSON serialization
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def find_best_2d_projection(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> Tuple[int, int, float]:
    """Find the 2D projection with best cluster separation (highest silhouette score)."""
    from itertools import combinations
    
    n_components = embedding.shape[1]
    if n_components < 2:
        return 0, 1, 0.0
    
    best_pair = (0, 1)
    best_score = -1.0
    
    for i, j in combinations(range(n_components), 2):
        emb_2d = embedding[:, [i, j]]
        try:
            sil_2d = silhouette_score(emb_2d, labels)
            if sil_2d > best_score:
                best_score = sil_2d
                best_pair = (i, j)
        except Exception:
            continue
    
    return best_pair[0], best_pair[1], best_score


def prepare_visualization_dataset(
    original_df: pd.DataFrame,
    labels: np.ndarray,
    embedding: np.ndarray,
    kmeans: KMeans,
) -> Tuple[pd.DataFrame, Dict]:
    if embedding.shape[1] < 2:
        raise ValueError("UMAP embedding must have at least 2 components for visualization")

    viz_df = original_df.reset_index(drop=True).copy()
    viz_df["cluster"] = labels.astype(int)
    
    # Store ALL UMAP components
    n_components = embedding.shape[1]
    for i in range(n_components):
        viz_df[f"umap_{i+1}"] = embedding[:, i].astype(float)
    
    # Find best 2D projection for visualization
    best_dim1, best_dim2, best_2d_score = find_best_2d_projection(embedding, labels)
    
    # Keep umap_x and umap_y as the BEST projection (for backward compatibility)
    viz_df["umap_x"] = embedding[:, best_dim1].astype(float)
    viz_df["umap_y"] = embedding[:, best_dim2].astype(float)
    viz_df["best_projection_dims"] = f"{best_dim1+1},{best_dim2+1}"

    patient_ids = None
    if "Student_No" in viz_df.columns:
        patient_ids = (
            viz_df["Student_No"].astype(str).str.strip().replace({"nan": "", "None": ""})
        )
    if patient_ids is None:
        patient_ids = pd.Series(["" for _ in range(len(viz_df))])
    missing_mask = patient_ids.eq("")
    patient_ids.loc[missing_mask] = (viz_df.index[missing_mask] + 1).astype(str)
    viz_df["patient_id"] = patient_ids

    if {"First_Name", "Last_Name"}.issubset(viz_df.columns):
        viz_df["display_name"] = (
            viz_df["First_Name"].fillna("").astype(str).str.strip() + " " +
            viz_df["Last_Name"].fillna("").astype(str).str.strip()
        ).str.strip()

    distances = kmeans.transform(embedding)
    inv_dist = 1.0 / (distances + 1e-9)
    confidence = inv_dist / inv_dist.sum(axis=1, keepdims=True)
    viz_df["cluster_confidence"] = confidence[np.arange(len(labels)), labels]

    risk_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    viz_df["risk_label"] = viz_df["cluster"].map(lambda c: risk_labels.get(int(c), f"Cluster {c}"))

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_patients": int(len(viz_df)),
        "n_umap_components": n_components,
        "best_2d_projection": {
            "dimensions": [best_dim1 + 1, best_dim2 + 1],
            "silhouette_2d": round(best_2d_score, 4),
        },
        "clusters": [],
    }
    total = len(viz_df)
    numeric_cols = viz_df.select_dtypes(include=[np.number]).columns
    for cluster_id, group in viz_df.groupby("cluster"):
        cluster_item = {
            "cluster": int(cluster_id),
            "count": int(len(group)),
            "percentage": round((len(group) / total) * 100, 2),
        }
        if "bmi" in group.columns:
            cluster_item["mean_bmi"] = round(pd.to_numeric(group["bmi"], errors="coerce").mean(skipna=True), 2)
        if "age_years" in group.columns:
            cluster_item["mean_age"] = round(pd.to_numeric(group["age_years"], errors="coerce").mean(skipna=True), 2)
        for col in numeric_cols:
            if col.startswith("umap_"):
                continue
            if col in ("cluster", "umap_x", "umap_y"):
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            if values.notna().any():
                cluster_item.setdefault("feature_means", {})[col] = round(float(values.mean()), 4)
        summary["clusters"].append(cluster_item)
    summary["clusters"].sort(key=lambda item: item["cluster"])

    return viz_df, summary


def export_visualization_payload(artifact_dir: Path, viz_df: pd.DataFrame, summary: Dict) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    viz_json_path = artifact_dir / "cluster_visualization.json"
    viz_csv_path = artifact_dir / "cluster_visualization.csv"
    summary_path = artifact_dir / "cluster_summary.json"

    viz_records = json.loads(viz_df.to_json(orient="records", date_format="iso"))
    with open(viz_json_path, "w", encoding="utf-8") as fh:
        json.dump(viz_records, fh, indent=2)
    viz_df.to_csv(viz_csv_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Saved visualization dataset with {len(viz_records)} records to {viz_json_path} and {viz_csv_path}"
    )
    print(f"Saved cluster summary to {summary_path}")

    return {
        "visualization_json": viz_json_path.name,
        "visualization_csv": viz_csv_path.name,
        "summary_json": summary_path.name,
    }


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data)
    original_df = df.copy()
    df_reduced, dropped_cols = drop_high_cardinality(df, args.high_card_threshold)
    df_encoded, freq_maps = frequency_encode(df_reduced)
    encoded_columns = df_encoded.columns.tolist()

    df_clean, X_scaled, prep_report, imputer, scaler, num_cols = clean_data(df_encoded)

    study = run_optuna(X_scaled, args.random_state, args.optuna_trials)
    best_params = study.best_params

    embedding, labels, reducer, kmeans, runtime = train_final_models(
        X_scaled, best_params, args.random_state
    )
    metrics = compute_metrics(embedding, labels, kmeans, runtime)

    viz_df, cluster_summary = prepare_visualization_dataset(original_df, labels, embedding, kmeans)
    viz_files = export_visualization_payload(args.artifacts, viz_df, cluster_summary)

    metadata = {
        "dataset_path": str(args.data),
        "random_state": args.random_state,
        "optuna_trials": args.optuna_trials,
        "best_params": best_params,
        "prep_report": prep_report,
        "dropped_columns": dropped_cols,
        "encoded_columns": encoded_columns,
        "numeric_columns": num_cols,
        "metrics": metrics,
        "cluster_summary": cluster_summary,
        "visualization_artifacts": viz_files,
    }

    save_artifacts(
        artifact_dir=args.artifacts,
        reducer=reducer,
        kmeans=kmeans,
        imputer=imputer,
        scaler=scaler,
        freq_maps=freq_maps,
        metadata=metadata,
    )
    # No auto-ID conversion — imputation handles missing identifiers


if __name__ == "__main__":
    main()
