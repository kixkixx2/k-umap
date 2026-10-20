# K-UMAP
https://k-umap.onrender.com/ 
## 📁 Project Structure

```
deploy/
├── app.py                 # Flask API server
├── train.py               # Model training script
├── build.py               # Build script for deployment
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── artifacts/             # Trained model artifacts
│   ├── umap_reducer.joblib
│   ├── kmeans_model.joblib
│   ├── knn_imputer.joblib
│   ├── scaler.joblib
│   └── cluster_visualization.json
├── data/
│   └── final_real.csv     # Training data
├── frontend/              # Static frontend files
│   ├── index.html
│   ├── analysis.html
│   ├── *.js
│   └── style.css
└── scripts/               # Utility scripts
```

## 🔧 Local Development

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
python train.py --data data/final_real.csv --artifacts artifacts --optuna-trials 50
```

### Run Server

```bash
# Development
python app.py

# Production (with gunicorn)
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```

### Access

- **Home Page**: http://localhost:5000/
- **Analysis Page**: http://localhost:5000/analysis.html
- **API Health**: http://localhost:5000/stats

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats` | GET | API health and statistics |
| `/patients` | GET | List all patients |
| `/predict` | POST | Predict cluster for new patient |
| `/cluster-data` | GET | Get cluster visualization data |

### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_years": 20,
    "year_level": 2,
    "gender": "Female",
    "bmi": 22.5,
    "has_respiratory_issue": 0,
    "has_pain": 1,
    "has_fever": 0,
    "has_allergy": 0,
    "is_uti": 0
  }'
```

## ⚠️ Important Notes

1. **Artifacts Required**: The `artifacts/` folder must contain trained models for the API to function
2. **Training on Deploy**: If artifacts are missing, the build script will automatically train the model
3. **Free Tier Limitations**: Render's free tier has cold starts (~30s) and limited resources

## 📝 License

MIT License
