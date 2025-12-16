"""Flask API for serving clustering artifacts and handling predictions."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS


class NaNSafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null for valid JSON output."""
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
    def encode(self, obj):
        return super().encode(self._sanitize(obj))
    
    def _sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize(obj.tolist())
        return obj

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Get the base directory (where app.py is located)
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", BASE_DIR / "artifacts"))
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", BASE_DIR / "frontend"))
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", 5000))

# Ensure artifact directory exists
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RISK_LABELS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
}

COLORABLE_FIELDS = [
    "age_years",
    "bmi",
    "year_level",
    "has_respiratory_issue",
    "has_pain",
    "has_fever",
    "has_allergy",
    "is_uti",
    "is_female",
    "cluster_confidence",
]

PLACEHOLDER_PATIENT_IDS = {
    "",
    "none",
    "null",
    "nan",
    "n/a",
    "not on form",
    "not available",
    "unknown",
}

DEFAULT_FIELD_INFO = {
    # Required fields for the ISU student health clustering model
    "age": {"type": "numeric", "required": True, "min": 1, "max": 120, "unit": "years"},
    "year_level": {"type": "numeric", "required": True, "min": 1, "max": 5, "unit": "year"},
    "gender": {"type": "categorical", "required": True, "options": ["Male", "Female", "Other"]},
    "BMI": {"type": "numeric", "required": False, "min": 10, "max": 60, "unit": "kg/m^2"},
    "school_id": {"type": "string", "required": True, "pattern": r"^\d{2}-\d{4}$", "example": "22-0641"},
    # Boolean diagnostic flags (optional, default to 0)
    "has_respiratory_issue": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    "has_pain": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    "has_fever": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    "has_allergy": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    "is_uti": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    # Optional fields (kept for backwards compatibility)
    "systolic_bp": {"type": "numeric", "required": False, "min": 70, "max": 250, "unit": "mmHg"},
    "diastolic_bp": {"type": "numeric", "required": False, "min": 40, "max": 150, "unit": "mmHg"},
    "heart_rate": {"type": "numeric", "required": False, "min": 40, "max": 200, "unit": "bpm"},
    "cholesterol_total": {"type": "numeric", "required": False, "min": 100, "max": 400, "unit": "mg/dL"},
    "blood_glucose": {"type": "numeric", "required": False, "min": 50, "max": 400, "unit": "mg/dL"},
    "doctor_visits_per_year": {"type": "numeric", "required": False, "min": 0, "max": 100, "unit": "visits"},
    "num_medications": {"type": "numeric", "required": False, "min": 0, "max": 50, "unit": "medications"},
    "medication_adherence": {"type": "numeric", "required": False, "min": 0, "max": 1, "unit": "ratio"},
    "treatment_success_rate": {"type": "numeric", "required": False, "min": 0, "max": 1, "unit": "ratio"},
    "ethnicity": {"type": "categorical", "required": False, "options": ["Caucasian", "African American", "Hispanic", "Asian", "Other"]},
    "insurance_type": {"type": "categorical", "required": False, "options": ["Private", "Medicare", "Medicaid", "Uninsured", "Other"]},
    "smoking_status": {"type": "categorical", "required": False, "options": ["Never", "Former", "Current"]},
    "alcohol_consumption": {"type": "categorical", "required": False, "options": ["None", "Light", "Moderate", "Heavy"]},
    "diabetes": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    "hypertension": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
    "heart_disease": {"type": "categorical", "required": False, "options": [0, 1, "0", "1"]},
}


# ---------------------------------------------------------------------------
# Artifact loader and prediction helpers
# ---------------------------------------------------------------------------
class ArtifactService:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.metadata = self._load_json("training_metadata.json")
        self.reducer = self._load_joblib("umap_reducer.joblib")
        self.kmeans = self._load_joblib("kmeans_model.joblib")
        self.imputer = self._load_joblib("knn_imputer.joblib")
        self.scaler = self._load_joblib("scaler.joblib")
        self.freq_maps = self._load_joblib("frequency_encoding_maps.joblib") or {}
        self.encoded_columns: List[str] = self.metadata.get("encoded_columns", [])
        self.cluster_summary = self._load_json("cluster_summary.json")
        self.cluster_profiles = _generate_cluster_profiles(self.cluster_summary)
        self._cluster_records = self._load_cluster_records()
        self._cluster_index = {str(rec.get("patient_id")): rec for rec in self._cluster_records}
        # Track file modification time for multi-worker synchronization
        self._cluster_file_mtime = self._get_cluster_file_mtime()

    def _get_cluster_file_mtime(self) -> float:
        """Get modification time of cluster visualization file."""
        cluster_path = self.artifact_dir / "cluster_visualization.json"
        try:
            return cluster_path.stat().st_mtime if cluster_path.exists() else 0.0
        except Exception:
            return 0.0

    def _refresh_if_stale(self) -> None:
        """Reload cluster records if file has been modified by another worker."""
        current_mtime = self._get_cluster_file_mtime()
        if current_mtime > self._cluster_file_mtime:
            print(f"[ArtifactService] Reloading cluster records (file modified by another worker)")
            self._cluster_records = self._load_cluster_records()
            self._cluster_index = {str(rec.get("patient_id")): rec for rec in self._cluster_records}
            self._cluster_file_mtime = current_mtime

    @property
    def ready(self) -> bool:
        return all([
            self.metadata,
            self.reducer is not None,
            self.kmeans is not None,
            self.imputer is not None,
            self.scaler is not None,
            bool(self._cluster_records),
        ])

    def list_patients(self) -> List[Dict[str, Any]]:
        self._refresh_if_stale()
        return self._cluster_records

    def get_patient(self, patient_id: str) -> Dict[str, Any] | None:
        if patient_id is None:
            return None
        self._refresh_if_stale()
        return self._cluster_index.get(str(patient_id))

    def add_patient(self, patient_record: Dict[str, Any]) -> bool:
        """Add a newly predicted patient to the cluster records and persist to JSON file."""
        try:
            # First, refresh from file in case another worker added patients
            self._refresh_if_stale()
            
            patient_id = str(patient_record.get("patient_id", ""))
            if not patient_id:
                return False
            
            # Normalize the record to match the format expected by the frontend
            normalized_record = self._normalize_patient_record(patient_record)
            
            # Check if patient already exists
            if patient_id in self._cluster_index:
                # Update existing patient
                existing_idx = next(
                    (i for i, rec in enumerate(self._cluster_records) if str(rec.get("patient_id")) == patient_id),
                    None
                )
                if existing_idx is not None:
                    self._cluster_records[existing_idx] = normalized_record
            else:
                # Add new patient
                self._cluster_records.append(normalized_record)
            
            # Update index
            self._cluster_index[patient_id] = normalized_record
            
            # Persist to file
            self._save_cluster_records()
            
            # Update our tracked mtime so we don't trigger unnecessary reloads
            self._cluster_file_mtime = self._get_cluster_file_mtime()
            return True
        except Exception as e:
            print(f"Error adding patient: {e}")
            return False

    def _normalize_patient_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a patient record to include all computed fields for frontend."""
        normalized = dict(record)
        
        # Ensure patient_id is set
        patient_id = normalized.get("patient_id") or normalized.get("Student_No")
        normalized["patient_id"] = patient_id
        normalized["auto_generated_id"] = str(patient_id).startswith("SIM-") if patient_id else True
        
        # Get cluster profile
        cluster_id = normalized.get("cluster")
        profile = None
        if cluster_id is not None and self.cluster_profiles:
            profile = self.cluster_profiles.get(int(cluster_id))
        if profile:
            normalized["cluster_label"] = profile.get("label")
            normalized["cluster_profile"] = {
                "risk_level": profile.get("risk_level"),
                "risk_summary": profile.get("risk_summary"),
                "care_focus": profile.get("care_focus"),
                "key_characteristics": profile.get("key_characteristics"),
                "recommendations": profile.get("recommendations"),
            }
        
        # Set coordinates
        x_value = _safe_float(normalized.get("umap_x"))
        y_value = _safe_float(normalized.get("umap_y"))
        normalized["x"] = x_value if x_value is not None else 0.0
        normalized["y"] = y_value if y_value is not None else 0.0
        
        # Set confidence
        confidence_value = _safe_float(normalized.get("cluster_confidence"))
        normalized["confidence"] = confidence_value if confidence_value is not None else None
        
        # Extract features
        features: Dict[str, float] = {}
        for field in COLORABLE_FIELDS:
            value = _safe_float(normalized.get(field))
            if value is not None:
                features[field] = value
        normalized["features"] = features
        
        return normalized

    def _save_cluster_records(self) -> None:
        """Save cluster records to JSON file."""
        cluster_path = self.artifact_dir / "cluster_visualization.json"
        # Create a backup first
        backup_path = self.artifact_dir / "cluster_visualization.json.bak"
        if cluster_path.exists():
            import shutil
            shutil.copy(cluster_path, backup_path)
        
        # Prepare records for saving (strip internal fields)
        save_records = []
        for rec in self._cluster_records:
            save_rec = {}
            for key, value in rec.items():
                # Skip computed fields that will be regenerated on load
                if key in ("cluster_label", "cluster_profile", "x", "y", "confidence", "features", "auto_generated_id"):
                    continue
                # Handle NaN values
                if isinstance(value, float) and (value != value):  # NaN check
                    save_rec[key] = None
                else:
                    save_rec[key] = value
            save_records.append(save_rec)
        
        with open(cluster_path, "w", encoding="utf-8") as fh:
            json.dump(save_records, fh, indent=2, cls=NaNSafeJSONEncoder)

    def predict_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if (
            not self.ready
            or self.reducer is None
            or self.kmeans is None
            or self.imputer is None
            or self.scaler is None
        ):
            raise RuntimeError("Artifacts are not fully loaded. Run train.py first.")

        features, sanitized_payload = self._prepare_features(payload)
        reducer = cast(Any, self.reducer)
        kmeans = cast(Any, self.kmeans)

        embedding = reducer.transform(features)
        cluster = int(kmeans.predict(embedding)[0])
        distances = kmeans.transform(embedding)
        inv_distance = 1.0 / (distances + 1e-9)
        confidences = inv_distance / inv_distance.sum(axis=1, keepdims=True)
        confidence = float(confidences[0, cluster])

        profile = self.cluster_profiles.get(cluster)

        coords = embedding[0]
        umap_x = float(coords[0])
        umap_y = float(coords[1]) if coords.shape[0] > 1 else 0.0

        patient_id = (
            payload.get("patient_id")
            or payload.get("Student_No")
            or sanitized_payload.get("Student_No")
            or f"SIM-{uuid.uuid4().hex[:8]}"
        )

        insights = build_insights(payload)
        risk_label = (profile or {}).get("label") or RISK_LABELS.get(cluster, f"Cluster {cluster}")

        return {
            "success": True,
            "patient_id": patient_id,
            "cluster": cluster,
            "confidence": confidence,
            "risk_label": risk_label,
            "umap_coordinates": {"x": umap_x, "y": umap_y},
            "patient_snapshot": payload,
            "sanitized_features": sanitized_payload,
            "insights": insights,
            "clinical_intelligence": build_clinical_intelligence(cluster, insights, profile),
            "cluster_profile": profile,
        }

    def batch_predict(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        successes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            try:
                result = self.predict_single(row)
                successes.append({
                    "row": idx,
                    "patient_id": result["patient_id"],
                    "cluster": result["cluster"],
                    "confidence": result["confidence"],
                    "umap_x": result["umap_coordinates"]["x"],
                    "umap_y": result["umap_coordinates"]["y"],
                    "input_data": row,
                })
            except Exception as exc:  # noqa: BLE001 - need to capture all issues
                failures.append({"row": idx, "errors": [str(exc)]})
        return successes, failures

    def _prepare_features(self, payload: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        frame = pd.DataFrame([payload]).copy()
        frame.columns = frame.columns.astype(str)
        frame = frame.replace({"": np.nan, "None": np.nan, "nan": np.nan})

        if not self.encoded_columns:
            raise RuntimeError("Metadata missing encoded column ordering.")

        for column in self.encoded_columns:
            if column not in frame:
                frame[column] = np.nan
        frame = frame[self.encoded_columns]

        for column, mapping in (self.freq_maps or {}).items():
            if column in frame:
                frame[column] = frame[column].map(mapping).fillna(0)

        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Preprocessing artifacts missing. Re-run training.")

        numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
        imputer = cast(Any, self.imputer)
        scaler = cast(Any, self.scaler)
        imputed = imputer.transform(numeric_frame)
        # Wrap with column names so StandardScaler sees the expected feature names
        imputed_df = pd.DataFrame(imputed, columns=numeric_frame.columns)
        scaled = scaler.transform(imputed_df)
        
        # Sanitize the output dict to remove NaN values
        raw_dict = numeric_frame.iloc[0].to_dict()
        sanitized_dict = {}
        for k, v in raw_dict.items():
            if v is None:
                sanitized_dict[k] = None
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                sanitized_dict[k] = None
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                if np.isnan(v) or np.isinf(v):
                    sanitized_dict[k] = None
                else:
                    sanitized_dict[k] = float(v)
            elif isinstance(v, (np.integer, np.int64, np.int32)):
                sanitized_dict[k] = int(v)
            else:
                sanitized_dict[k] = v
        
        return scaled, sanitized_dict

    def _load_cluster_records(self) -> List[Dict[str, Any]]:
        cluster_path = self.artifact_dir / "cluster_visualization.json"
        if cluster_path.exists():
            with open(cluster_path, "r", encoding="utf-8") as fh:
                original_records = json.load(fh)
                normalized: List[Dict[str, Any]] = []
                for idx, record in enumerate(original_records, start=1):
                    normalized_record = dict(record)
                    patient_identifier = normalized_record.get("patient_id") or normalized_record.get("Student_No")
                    normalized_id, auto_generated = _normalize_patient_identifier(patient_identifier, idx)
                    normalized_record["patient_id"] = normalized_id
                    normalized_record["auto_generated_id"] = auto_generated

                    cluster_id = normalized_record.get("cluster")
                    profile = None
                    if cluster_id is not None and self.cluster_profiles:
                        profile = self.cluster_profiles.get(int(cluster_id))
                    if profile:
                        normalized_record["cluster_label"] = profile.get("label")
                        normalized_record["cluster_profile"] = {
                            "risk_level": profile.get("risk_level"),
                            "risk_summary": profile.get("risk_summary"),
                            "care_focus": profile.get("care_focus"),
                            "key_characteristics": profile.get("key_characteristics"),
                            "recommendations": profile.get("recommendations"),
                        }

                    x_value = _safe_float(normalized_record.get("umap_x"))
                    y_value = _safe_float(normalized_record.get("umap_y"))
                    normalized_record["x"] = x_value if x_value is not None else 0.0
                    normalized_record["y"] = y_value if y_value is not None else 0.0
                    confidence_value = _safe_float(normalized_record.get("cluster_confidence"))
                    normalized_record["confidence"] = confidence_value if confidence_value is not None else None

                    features: Dict[str, float] = {}
                    for field in COLORABLE_FIELDS:
                        value = _safe_float(normalized_record.get(field))
                        if value is not None:
                            features[field] = value
                    normalized_record["features"] = features

                    normalized.append(normalized_record)
                return normalized
        return []

    def _load_json(self, filename: str) -> Dict[str, Any]:
        path = self.artifact_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_joblib(self, filename: str):
        path = self.artifact_dir / filename
        if not path.exists():
            return None
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Insight helpers
# ---------------------------------------------------------------------------
def build_insights(payload: Dict[str, Any]) -> Dict[str, str] | None:
    if not payload:
        return None
    insights: Dict[str, str] = {}
    bmi = _safe_float(payload.get("BMI") or payload.get("bmi"))
    if bmi is not None:
        insights["bmi_category"] = categorize_bmi(bmi)
    systolic = _safe_float(payload.get("systolic_bp"))
    diastolic = _safe_float(payload.get("diastolic_bp"))
    if systolic is not None and diastolic is not None:
        insights["bp_category"] = categorize_bp(systolic, diastolic)
    glucose = _safe_float(payload.get("blood_glucose"))
    if glucose is not None:
        insights["diabetes_risk"] = categorize_glucose(glucose)
    return insights or None


def build_clinical_intelligence(
    cluster: int,
    insights: Dict[str, str] | None,
    profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    default_recommendations = {
        0: "Routine preventive care recommended.",
        1: "Schedule follow-up and reinforce lifestyle coaching.",
        2: "Consider escalated monitoring and multidisciplinary review.",
    }
    alerts = []
    if insights:
        if insights.get("bp_category") in {"Stage 2 Hypertension", "Hypertensive Crisis"}:
            alerts.append("Blood pressure above target range.")
        if insights.get("diabetes_risk") == "High Risk":
            alerts.append("Elevated glucose levels detected.")
    summary = default_recommendations.get(cluster, "Review patient context for personalized guidance.")
    recs = []
    if profile:
        summary = profile.get("care_focus") or profile.get("risk_summary") or summary
        recs = profile.get("recommendations", [])
    payload = {
        "summary": summary,
        "alerts": alerts,
        "recommendations": recs,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    return payload


def categorize_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    if bmi < 35:
        return "Obesity Class I"
    if bmi < 40:
        return "Obesity Class II"
    return "Obesity Class III"


def categorize_bp(systolic: float, diastolic: float) -> str:
    if systolic < 120 and diastolic < 80:
        return "Normal"
    if systolic < 130 and diastolic < 80:
        return "Elevated"
    if systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    if systolic < 180 or diastolic < 120:
        return "Stage 2 Hypertension"
    return "Hypertensive Crisis"


def categorize_glucose(glucose: float) -> str:
    if glucose < 100:
        return "Low Risk"
    if glucose < 126:
        return "Pre-Diabetic"
    return "High Risk"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _get_field_value(row: Dict[str, Any], *field_names: str) -> Any:
    """Get value from row using multiple possible field names (case-insensitive)."""
    row_lower = {k.lower(): v for k, v in row.items()}
    for name in field_names:
        # Try exact match first
        if name in row:
            return row[name]
        # Try lowercase match
        if name.lower() in row_lower:
            return row_lower[name.lower()]
    return None


def validate_batch_row(row: Dict[str, Any], row_num: int) -> List[str]:
    """Validate a single row from batch CSV upload.
    
    Returns a list of error messages. Empty list means valid.
    """
    errors = []
    
    # 1. Validate Student_No format: xx-xxxx (2 digits, dash, 4 digits)
    # Also handle cases where Excel might have mangled the format (e.g., 1-641 instead of 01-0641)
    student_id = _get_field_value(row, "Student_No", "school_id", "student_no", "StudentNo", "student_id")
    student_id = str(student_id or "").strip()
    if not student_id:
        errors.append(f"Row {row_num}: Student_No/school_id is required")
    else:
        # Try to normalize Student_No format (handle Excel stripping leading zeros)
        pattern = r"^\d{2}-\d{4}$"
        if not re.match(pattern, student_id):
            # Try to fix common Excel issues: "1-641" -> "01-0641"
            parts = student_id.split("-")
            if len(parts) == 2:
                try:
                    normalized = f"{int(parts[0]):02d}-{int(parts[1]):04d}"
                    if re.match(pattern, normalized):
                        student_id = normalized  # Use normalized version
                        row["Student_No"] = normalized  # Update the row too
                    else:
                        errors.append(f"Row {row_num}: Student_No '{student_id}' must be in format XX-XXXX (e.g., 22-0641)")
                except ValueError:
                    errors.append(f"Row {row_num}: Student_No '{student_id}' must be in format XX-XXXX (e.g., 22-0641)")
            else:
                errors.append(f"Row {row_num}: Student_No '{student_id}' must be in format XX-XXXX (e.g., 22-0641)")
    
    # 1b. Validate First_Name and Last_Name (required for records, but hidden in display for privacy)
    first_name = _get_field_value(row, "First_Name", "first_name", "FirstName", "firstname")
    if not first_name or str(first_name).strip() == "":
        errors.append(f"Row {row_num}: First_Name is required")
    
    last_name = _get_field_value(row, "Last_Name", "last_name", "LastName", "lastname")
    if not last_name or str(last_name).strip() == "":
        errors.append(f"Row {row_num}: Last_Name is required")
    
    # 2. Validate year_level (required, must be 1-5)
    year_level = _get_field_value(row, "year_level", "Year_Level", "yearlevel", "year")
    if year_level is None or year_level == "":
        errors.append(f"Row {row_num}: year_level is required")
    else:
        year_val = _safe_float(year_level)
        if year_val is None:
            errors.append(f"Row {row_num}: year_level '{year_level}' must be a number")
        elif year_val < 1 or year_val > 5:
            errors.append(f"Row {row_num}: year_level must be between 1 and 5, got {year_val}")
    
    # 3. Validate age_years (required, must be reasonable age)
    age = _get_field_value(row, "age_years", "age", "Age", "Age_Years")
    if age is None or age == "":
        errors.append(f"Row {row_num}: age_years is required")
    else:
        age_val = _safe_float(age)
        if age_val is None:
            errors.append(f"Row {row_num}: age_years '{age}' must be a number")
        elif age_val < 15 or age_val > 60:
            errors.append(f"Row {row_num}: age_years must be between 15 and 60, got {age_val}")
    
    # 4. Validate gender/is_female (required)
    gender = _get_field_value(row, "gender", "Gender", "is_female", "sex", "Sex")
    if gender is None or gender == "":
        errors.append(f"Row {row_num}: gender or is_female is required")
    else:
        gender_str = str(gender).strip().lower()
        valid_genders = ["male", "female", "m", "f", "0", "1"]
        if gender_str not in valid_genders:
            errors.append(f"Row {row_num}: gender '{gender}' must be Male/Female or 0/1")
    
    # 5. Validate height_cm and weight_kg (required for BMI calculation)
    height = _get_field_value(row, "height_cm", "height", "Height", "Height_cm")
    weight = _get_field_value(row, "weight_kg", "weight", "Weight", "Weight_kg")
    bmi = _get_field_value(row, "bmi", "BMI", "Bmi")
    
    # If BMI is provided directly, validate it
    if bmi is not None and bmi != "":
        bmi_val = _safe_float(bmi)
        if bmi_val is not None and (bmi_val < 10 or bmi_val > 60):
            errors.append(f"Row {row_num}: bmi must be between 10 and 60, got {bmi_val}")
    else:
        # BMI not provided, require height and weight
        if height is None or height == "":
            errors.append(f"Row {row_num}: height_cm is required (or provide BMI directly)")
        else:
            height_val = _safe_float(height)
            if height_val is None:
                errors.append(f"Row {row_num}: height_cm '{height}' must be a number")
            elif height_val < 100 or height_val > 250:
                errors.append(f"Row {row_num}: height_cm must be between 100 and 250, got {height_val}")
        
        if weight is None or weight == "":
            errors.append(f"Row {row_num}: weight_kg is required (or provide BMI directly)")
        else:
            weight_val = _safe_float(weight)
            if weight_val is None:
                errors.append(f"Row {row_num}: weight_kg '{weight}' must be a number")
            elif weight_val < 30 or weight_val > 200:
                errors.append(f"Row {row_num}: weight_kg must be between 30 and 200, got {weight_val}")
    
    # 6. Validate diagnosis flags (required, must be 0/1 or boolean-like)
    # Support multiple column name variations for each flag
    diagnosis_flags = {
        "has_respiratory_issue": ["has_respiratory_issue", "has_respir", "respiratory", "resp_issue", "has_respiratory"],
        "has_pain": ["has_pain", "pain", "Pain"],
        "has_fever": ["has_fever", "fever", "Fever"],
        "has_allergy": ["has_allergy", "allergy", "Allergy", "allergies"],
        "is_uti": ["is_uti", "uti", "UTI", "has_uti"],
    }
    
    for flag_name, variants in diagnosis_flags.items():
        value = _get_field_value(row, *variants)
        if value is None or value == "":
            errors.append(f"Row {row_num}: {flag_name} is required")
        else:
            # Handle various formats: 0, 1, "0", "1", True, False, "true", "false", etc.
            val_str = str(value).strip().lower()
            valid_string_values = ["0", "1", "0.0", "1.0", "true", "false", "yes", "no"]
            if val_str not in valid_string_values:
                errors.append(f"Row {row_num}: {flag_name} '{value}' must be 0/1 or true/false")
    
    return errors


def preprocess_form_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert frontend form data to model-expected feature columns.
    
    The model expects: has_respiratory_issue, has_pain, has_fever, has_allergy,
    is_uti, year_level, age_years, age_group, is_female, bmi
    
    The form sends: BMI or (height_cm + weight_kg), gender, age, year_level, and boolean flags.
    """
    processed = dict(payload)  # Copy to avoid mutating original
    
    # Map school_id -> Student_No and patient_id (use as patient identifier)
    student_id = _get_field_value(processed, "school_id", "Student_No", "student_no", "StudentNo", "student_id")
    if student_id:
        school_id = str(student_id).strip()
        if school_id:
            processed["Student_No"] = school_id
            processed["patient_id"] = school_id
    
    # Calculate BMI from height_cm and weight_kg if BMI not provided
    bmi_val = _get_field_value(processed, "BMI", "bmi", "Bmi")
    height_val = _get_field_value(processed, "height_cm", "height", "Height", "Height_cm")
    weight_val = _get_field_value(processed, "weight_kg", "weight", "Weight", "Weight_kg")
    
    if (bmi_val is None or bmi_val == "") and height_val is not None and weight_val is not None:
        height_cm = _safe_float(height_val)
        weight_kg = _safe_float(weight_val)
        if height_cm and weight_kg and height_cm > 0:
            height_m = height_cm / 100.0
            calculated_bmi = weight_kg / (height_m * height_m)
            processed["bmi"] = round(calculated_bmi, 1)
            # Store height and weight for reference
            processed["height_cm"] = height_cm
            processed["weight_kg"] = weight_kg
    elif bmi_val is not None and "bmi" not in processed:
        processed["bmi"] = _safe_float(bmi_val)
    
    # Map age -> age_years
    age_val = _get_field_value(processed, "age", "age_years", "Age", "Age_Years")
    if age_val is not None and "age_years" not in processed:
        processed["age_years"] = _safe_float(age_val)
    
    # Map gender -> is_female (1 if Female, 0 otherwise)
    gender_val = _get_field_value(processed, "gender", "Gender", "sex", "Sex")
    if gender_val is not None and "is_female" not in processed:
        gender = str(gender_val).strip().lower()
        processed["is_female"] = 1 if gender == "female" else 0
    
    # Derive age_group from age_years if not present
    age_years = _safe_float(processed.get("age_years") or processed.get("age"))
    if age_years is not None and "age_group" not in processed:
        if age_years < 18:
            processed["age_group"] = "<18"
        elif age_years <= 21:
            processed["age_group"] = "18-21"
        else:
            processed["age_group"] = "22+"
    
    # Map diagnosis flag variations to expected column names
    flag_mappings = {
        "has_respiratory_issue": ["has_respiratory_issue", "has_respir", "respiratory", "resp_issue", "has_respiratory"],
        "has_pain": ["has_pain", "pain", "Pain"],
        "has_fever": ["has_fever", "fever", "Fever"],
        "has_allergy": ["has_allergy", "allergy", "Allergy", "allergies"],
        "is_uti": ["is_uti", "uti", "UTI", "has_uti"],
    }
    
    for target_name, variants in flag_mappings.items():
        if target_name not in processed or processed.get(target_name) in (None, ""):
            val = _get_field_value(processed, *variants)
            if val is not None:
                processed[target_name] = val
    
    # Ensure boolean flags are numeric (0 or 1)
    bool_flags = ["has_respiratory_issue", "has_pain", "has_fever", "has_allergy", "is_uti"]
    for flag in bool_flags:
        val = processed.get(flag)
        if val is None or val == "":
            processed[flag] = 0
        elif isinstance(val, bool):
            processed[flag] = 1 if val else 0
        elif isinstance(val, str):
            processed[flag] = 1 if val.lower() in ("1", "true", "yes", "on") else 0
        else:
            processed[flag] = 1 if val else 0
    
    # Default year_level if missing
    year_val = _get_field_value(processed, "year_level", "Year_Level", "yearlevel", "year")
    if year_val is not None:
        processed["year_level"] = _safe_float(year_val) or 1
    elif "year_level" not in processed or processed.get("year_level") in (None, ""):
        processed["year_level"] = 1  # Default to first year
    
    return processed


def _normalize_patient_identifier(raw_id: Any, fallback_index: int) -> Tuple[str, bool]:
    if raw_id is not None:
        candidate = str(raw_id).strip()
    else:
        candidate = ""

    if candidate:
        normalized = candidate.strip()
        low = normalized.lower()
        # If explicit placeholder value, treat as missing
        if low in PLACEHOLDER_PATIENT_IDS:
            pass
        # If candidate already follows school ID format (e.g. 22-0641), keep it
        elif re.fullmatch(r"\d{2}-\d{4}", normalized):
            return normalized, False
        # If candidate already is an auto ID we produced earlier, keep and mark auto
        elif re.fullmatch(r"(?i)auto\s00-\d{4}", normalized):
            return normalized, True

    # Generate auto IDs in school format prefixed with 'auto '
    # Example: auto 00-0001, auto 00-0002 ...
    return (f"auto 00-{fallback_index:04d}", True)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "0%"
    return f"{value * 100:.0f}%"


def _format_whole_number(value: float | None) -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"{int(round(value))}"


def _describe_cluster_profile(entry: Dict[str, Any]) -> Dict[str, Any]:
    cluster_id = entry.get("cluster")
    features = entry.get("feature_means", {}) or {}
    resp = float(features.get("has_respiratory_issue", 0.0) or 0.0)
    fever = float(features.get("has_fever", 0.0) or 0.0)
    pain = float(features.get("has_pain", 0.0) or 0.0)
    allergy = float(features.get("has_allergy", 0.0) or 0.0)
    uti = float(features.get("is_uti", 0.0) or 0.0)
    bmi = float(entry.get("mean_bmi") or features.get("bmi") or 0.0)
    age = float(entry.get("mean_age") or features.get("age_years") or 0.0)
    female_ratio = float(features.get("is_female", 0.0) or 0.0)
    year_level = float(features.get("year_level", 0.0) or 0.0)
    population_share = entry.get("percentage", 0.0)

    symptom_burden = (resp + fever + pain + allergy + uti) / 5.0
    high_resp = resp >= 0.8 and fever >= 0.8
    wellness = symptom_burden <= 0.05
    metabolic = bmi >= 30 and not wellness

    female_pct = _format_percent(female_ratio)
    resp_pct = _format_percent(resp)
    fever_pct = _format_percent(fever)
    pain_pct = _format_percent(pain)
    allergy_pct = _format_percent(allergy)

    age_display = _format_whole_number(age)
    year_level_display = _format_whole_number(year_level)

    if high_resp:
        label = "Acute Respiratory-Febrile Cohort"
        risk_summary = "Students presenting with concurrent fever and respiratory complaints—treat as an infectious respiratory cluster."
        care_focus = "Prioritize respiratory infection protocols, hydration, and early warning monitoring for deterioration."
        key_characteristics = [
            f"{resp_pct} report respiratory difficulty on triage",
            f"{fever_pct} arrive febrile requiring antipyretic management",
            f"{female_pct} female; mean BMI {bmi:.1f} with lean body habitus",
        ]
        recommendations = [
            "Implement rapid respiratory assessment (respiratory rate, pulse oximetry, lung exam).",
            "Initiate fever management bundle and review exposure/contagion controls for shared housing.",
            "Escalate to tele-infectious-disease consult when symptoms persist beyond 48 hours or red flags emerge.",
        ]
        risk_level = "Acute infectious risk"
    elif wellness:
        label = "Preventive / Clearance Cohort"
        risk_summary = "Students seen for wellness, clearance, or vitals tracking with no consistent acute findings."
        care_focus = "Maintain preventive services and readiness for emerging complaints rather than acute intervention."
        key_characteristics = [
            f"Mean BMI {bmi:.1f} within the normal range",
            f"{female_pct} female; population skewed toward first-year level {year_level_display}",
            "No recurring respiratory, fever, pain, allergy, or UTI signals detected",
        ]
        recommendations = [
            "Continue annual comprehensive screening (mental health, vaccination status, lifestyle coaching).",
            "Use visits to reinforce nutrition, sleep hygiene, and exercise adherence.",
            "Deploy early-warning education so students report new symptoms promptly.",
        ]
        risk_level = "Preventive focus"
    elif metabolic:
        label = "Metabolic-Inflammatory Support Cohort"
        risk_summary = "High-BMI students with intermittent respiratory, pain, and febrile episodes—needs cardiometabolic management plus symptom surveillance."
        care_focus = "Blend weight-management counseling with monitoring for inflammatory flares and infectious triggers."
        key_characteristics = [
            f"Mean BMI {bmi:.1f} (obesity class II range) with mean age {age_display} years",
            f"{resp_pct} respiratory complaints and {fever_pct} febrile encounters",
            f"{pain_pct} report pain and {allergy_pct} have allergy triggers",
        ]
        recommendations = [
            "Initiate nutrition, physical activity, and behavioral coaching pathways focused on cardiometabolic risk mitigation.",
            "Screen for hypertension, dyslipidemia, insulin resistance, and sleep-disordered breathing at each follow-up.",
            "Provide action plans for managing febrile or respiratory flares, including rapid follow-up slots.",
        ]
        risk_level = "Chronic metabolic risk"
    else:
        label = "Mixed Symptom Monitoring Cohort"
        risk_summary = "Students with mild-to-moderate symptom burden that oscillates across visits."
        care_focus = "Track symptom triggers and intervene early when complaints cluster."
        key_characteristics = [
            f"{resp_pct} respiratory and {fever_pct} febrile presentations",
            f"Mean BMI {bmi:.1f}; {female_pct} female",
            f"Pain reported in {pain_pct} of encounters",
        ]
        recommendations = [
            "Structure symptom diaries and digital check-ins to capture flare patterns.",
            "Coordinate multi-disciplinary reviews (primary care, behavioral health, nutrition) when clusters emerge.",
            "Apply targeted labs or imaging if symptoms persist beyond standard observation windows.",
        ]
        risk_level = "Variable risk"

    age_summary = f" and mean age {age_display} years" if age_display != "N/A" else ""
    summary = (
        f"Cluster {cluster_id} covers {population_share:.2f}% of the cohort with mean BMI {bmi:.1f}{age_summary}."
    )

    return {
        "cluster": cluster_id,
        "label": label,
        "risk_level": risk_level,
        "risk_summary": risk_summary,
        "summary": summary,
        "care_focus": care_focus,
        "key_characteristics": key_characteristics,
        "recommendations": recommendations,
        "metrics": {
            "respiratory": resp,
            "fever": fever,
            "pain": pain,
            "allergy": allergy,
            "uti": uti,
            "bmi": bmi,
            "age": age,
            "female_ratio": female_ratio,
        },
    }


def _generate_cluster_profiles(cluster_summary: Dict[str, Any] | None) -> Dict[int, Dict[str, Any]]:
    if not cluster_summary:
        return {}
    profiles: Dict[int, Dict[str, Any]] = {}
    for entry in cluster_summary.get("clusters", []) or []:
        profile = _describe_cluster_profile(entry)
        if profile.get("cluster") is not None:
            profiles[int(profile["cluster"])] = profile
    return profiles


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization, converting NaN/Inf to None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    artifact_service = ArtifactService(ARTIFACT_DIR)
    frontend_root = FRONTEND_DIR.resolve()

    @app.get("/api")
    def api_index():
        base_url = request.host_url.rstrip("/")
        return jsonify(
            {
                "message": "ClusterMed API is running",
                "health": f"{base_url}/health",
                "endpoints": {
                    "predict": f"{base_url}/predict",
                    "batch_predict": f"{base_url}/api/batch_predict",
                    "patients": f"{base_url}/api/get_all_patients",
                    "patient_detail": f"{base_url}/api/get_patient/<patient_id>",
                    "field_info": f"{base_url}/api/field_info",
                    "cluster_summary": f"{base_url}/api/cluster_summary",
                },
                "instructions": "Use /predict for single predictions or /api/get_all_patients for visualization data.",
            }
        )

    @app.get("/health")
    def health_check():
        status = "healthy" if artifact_service.ready else "degraded"
        return jsonify({
            "status": status,
            "total_patients": len(artifact_service.list_patients()),
            "artifact_dir": str(ARTIFACT_DIR.resolve()),
        })

    @app.get("/api/field_info")
    def field_info():
        return jsonify({"success": True, "fields": DEFAULT_FIELD_INFO})

    @app.get("/api/get_all_patients")
    def get_all_patients():
        return jsonify(artifact_service.list_patients())

    @app.get("/api/get_patient/<patient_id>")
    def get_patient(patient_id: str):
        patient = artifact_service.get_patient(patient_id)
        if not patient:
            return jsonify({"success": False, "error": "Patient not found"}), 404
        return jsonify({"success": True, "patient": patient})

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        if not payload:
            return jsonify({"success": False, "error": "Request body must be JSON."}), 400
        try:
            # Preprocess form data to model-expected columns
            processed_payload = preprocess_form_payload(payload)
            result = artifact_service.predict_single(processed_payload)
            
            # Add the new patient to the cluster records and persist
            if result.get("success"):
                patient_record = {
                    "patient_id": result.get("patient_id"),
                    "Student_No": result.get("patient_id"),
                    "cluster": result.get("cluster"),
                    "cluster_confidence": result.get("confidence"),
                    "umap_x": result.get("umap_coordinates", {}).get("x", 0.0),
                    "umap_y": result.get("umap_coordinates", {}).get("y", 0.0),
                    **result.get("sanitized_features", {})
                }
                artifact_service.add_patient(patient_record)
            
            # Sanitize result to remove any NaN/Inf values before JSON encoding
            sanitized_result = _sanitize_for_json(result)
            return jsonify(sanitized_result)
        except Exception as exc:  # noqa: BLE001 - return error to client
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 500

    @app.post("/api/batch_predict")
    def batch_predict():
        if "file" not in request.files:
            return jsonify({"error": "CSV file missing. Upload via 'file' form field."}), 400
        file_storage = request.files["file"]
        try:
            file_storage.stream.seek(0)
            df = pd.read_csv(file_storage.stream)
        except Exception as exc:  # noqa: BLE001 - invalid CSV
            return jsonify({"error": f"Unable to read CSV: {exc}"}), 400

        rows = cast(List[Dict[str, Any]], df.to_dict(orient="records"))
        
        # Debug: Print received columns and first row
        print(f"[DEBUG] Batch upload - Columns: {list(df.columns)}")
        if rows:
            print(f"[DEBUG] First row: {rows[0]}")
        
        # Validate and filter rows
        validated_rows = []
        validation_errors = []
        skipped_duplicates = []
        existing_ids = {r.get("patient_id") or r.get("Student_No") for r in artifact_service._cluster_records}
        seen_ids_in_batch = set()
        
        for idx, row in enumerate(rows, start=1):
            errors = validate_batch_row(row, idx)
            if errors:
                print(f"[DEBUG] Row {idx} validation errors: {errors}")
            student_id = str(row.get("Student_No", "") or row.get("school_id", "")).strip()
            
            # Check for duplicate in existing records
            if student_id and student_id in existing_ids:
                skipped_duplicates.append({
                    "row": idx,
                    "student_id": student_id,
                    "reason": "Student ID already exists in the system"
                })
                continue
            
            # Check for duplicate within the batch
            if student_id and student_id in seen_ids_in_batch:
                skipped_duplicates.append({
                    "row": idx,
                    "student_id": student_id,
                    "reason": "Duplicate Student ID within this batch"
                })
                continue
            
            if student_id:
                seen_ids_in_batch.add(student_id)
            
            if errors:
                validation_errors.append({"row": idx, "errors": errors, "data": row})
            else:
                # Preprocess the row before prediction
                processed_row = preprocess_form_payload(row)
                validated_rows.append(processed_row)
        
        if not validated_rows:
            return jsonify({
                "error": "No valid rows to process.",
                "validation_errors": validation_errors,
                "skipped_duplicates": skipped_duplicates,
                "total_rows": len(rows),
                "invalid_count": len(validation_errors),
                "duplicate_count": len(skipped_duplicates)
            }), 400

        successes, failures = artifact_service.batch_predict(validated_rows)
        
        # Build clinical intelligence for each success and save to cluster records
        saved_count = 0
        for success in successes:
            cluster = success.get("cluster")
            profile = artifact_service.cluster_profiles.get(cluster) if artifact_service.cluster_profiles else None
            success["clinical_intelligence"] = build_clinical_intelligence(cluster, None, profile)
            success["cluster_profile"] = profile
            
            # Build patient record for persistence (to make them searchable)
            input_data = success.get("input_data", {})
            patient_record = {
                "patient_id": success.get("patient_id"),
                "Student_No": success.get("patient_id"),
                "cluster": cluster,
                "cluster_confidence": success.get("confidence"),
                "umap_x": success.get("umap_x"),
                "umap_y": success.get("umap_y"),
                # Include all input features
                "year_level": input_data.get("year_level"),
                "age_years": input_data.get("age_years"),
                "age_group": input_data.get("age_group"),
                "is_female": input_data.get("is_female"),
                "bmi": input_data.get("bmi"),
                "has_respiratory_issue": input_data.get("has_respiratory_issue"),
                "has_pain": input_data.get("has_pain"),
                "has_fever": input_data.get("has_fever"),
                "has_allergy": input_data.get("has_allergy"),
                "is_uti": input_data.get("is_uti"),
            }
            
            # Add patient to cluster records (makes them searchable)
            if artifact_service.add_patient(patient_record):
                saved_count += 1
        
        payload = {
            "total_rows": len(rows),
            "total_patients": len(validated_rows),
            "successful": len(successes),
            "saved_to_database": saved_count,
            "failed": len(failures),
            "invalid_count": len(validation_errors),
            "duplicate_count": len(skipped_duplicates),
            "results": successes,
        }
        if failures:
            payload["processing_errors"] = failures
        if validation_errors:
            payload["validation_errors"] = validation_errors
        if skipped_duplicates:
            payload["skipped_duplicates"] = skipped_duplicates
            
        return jsonify(payload)

    @app.get("/api/cluster_summary")
    def cluster_summary():
        if not artifact_service.cluster_summary:
            return jsonify({"success": False, "error": "Cluster summary unavailable."}), 404
        summary = artifact_service.cluster_summary
        return jsonify({
            "success": True,
            "generated_at": summary.get("generated_at"),
            "total_patients": summary.get("total_patients"),
            "clusters": summary.get("clusters", []),
            "profiles": artifact_service.cluster_profiles,
            "summary": summary,
        })

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path: str):
        if path.startswith("api"):
            abort(404)
        safe_path = path.strip("/") or "index.html"
        full_path = (frontend_root / safe_path).resolve()
        try:
            full_path.relative_to(frontend_root)
        except ValueError:
            abort(403)
        if full_path.is_dir():
            full_path = full_path / "index.html"
        if full_path.exists():
            relative_path = full_path.relative_to(frontend_root)
            return send_from_directory(str(frontend_root), str(relative_path))
        index_path = frontend_root / "index.html"
        if index_path.exists():
            return send_from_directory(str(frontend_root), "index.html")
        abort(404)

    return app


app = create_app()


if __name__ == "__main__":
    # Only enable debug mode in development
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=debug_mode, use_reloader=False)

