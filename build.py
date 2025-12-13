#!/usr/bin/env python3
"""Build script for Render deployment.

This script runs during the build phase to:
1. Train the model and generate artifacts (if not present)
2. Validate that all required artifacts exist
"""

import subprocess
import sys
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


def check_artifacts_exist():
    """Check if all required artifacts exist."""
    missing = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (ARTIFACT_DIR / artifact).exists():
            missing.append(artifact)
    return missing


def run_training():
    """Run the training script to generate artifacts."""
    print("=" * 60)
    print("Running model training...")
    print("=" * 60)
    
    # Find the data file
    data_file = DATA_DIR / "final_real.csv"
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return False
    
    # Run training
    cmd = [
        sys.executable, 
        str(BASE_DIR / "train.py"),
        "--data", str(data_file),
        "--artifacts", str(ARTIFACT_DIR),
        "--optuna-trials", "20",  # Reduced for faster build
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    
    if result.returncode != 0:
        print(f"ERROR: Training failed with code {result.returncode}")
        return False
    
    print("Training completed successfully!")
    return True


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
        print(f"Missing artifacts: {missing}")
        print("Will run training to generate artifacts...")
        
        if not run_training():
            print("ERROR: Failed to generate artifacts")
            sys.exit(1)
        
        # Verify artifacts were created
        missing = check_artifacts_exist()
        if missing:
            print(f"ERROR: Still missing artifacts after training: {missing}")
            sys.exit(1)
    else:
        print("All required artifacts found!")
    
    print("=" * 60)
    print("Build completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
