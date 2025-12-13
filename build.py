#!/usr/bin/env python3
"""Build script for Render deployment.

This script runs during the build phase to:
1. Validate that all required artifacts exist (from repo)
2. Skip training since artifacts are pre-generated from notebook

NOTE: Artifacts are pre-generated from K-UMAP.ipynb notebook with optimal
hyperparameters (8D UMAP, 3 clusters, silhouette=0.7249). Do NOT regenerate!
"""

import subprocess
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

REQUIRED_ARTIFACTS = [
    "umap_reducer.joblib",
    "kmeans_model.joblib", 
    "knn_imputer.joblib",
    "scaler.joblib",
    "cluster_visualization.json",
]

# Expected cluster count from notebook (do not change unless notebook is re-run)
EXPECTED_CLUSTERS = 3


def check_artifacts_exist():
    """Check if all required artifacts exist."""
    missing = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (ARTIFACT_DIR / artifact).exists():
            missing.append(artifact)
    return missing


def validate_cluster_count():
    """Validate that artifacts have the expected number of clusters."""
    cluster_summary_path = ARTIFACT_DIR / "cluster_summary.json"
    if cluster_summary_path.exists():
        try:
            with open(cluster_summary_path, 'r') as f:
                summary = json.load(f)
            actual_clusters = len(summary)
            print(f"Artifact validation: Found {actual_clusters} clusters (expected {EXPECTED_CLUSTERS})")
            return actual_clusters == EXPECTED_CLUSTERS
        except Exception as e:
            print(f"Warning: Could not validate cluster count: {e}")
    return True  # Skip validation if file doesn't exist


def main():
    """Main build function."""
    print("=" * 60)
    print("Patient Clustering API - Build Script")
    print("=" * 60)
    
    # Ensure artifact directory exists
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if artifacts exist
    missing = check_artifacts_exist()
    
    if missing:
        print(f"ERROR: Missing required artifacts: {missing}")
        print("Artifacts should be pre-generated from K-UMAP.ipynb notebook.")
        print("Please run the notebook and commit artifacts to the repository.")
        sys.exit(1)
    else:
        print("All required artifacts found!")
    
    # Validate cluster count matches expected
    if not validate_cluster_count():
        print(f"WARNING: Cluster count mismatch! Expected {EXPECTED_CLUSTERS} clusters.")
        print("This may indicate stale artifacts. Consider clearing Render disk.")
    
    print("=" * 60)
    print("Build completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
