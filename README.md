<div align="center">

# 🛵 Swiggy Delivery Time Prediction

**End-to-end MLOps pipeline predicting food delivery time from restaurant/delivery coordinates, weather, traffic, and vehicle features.**

![Python](https://img.shields.io/badge/-Python%203.12-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-006ACC?style=flat-square)
![LightGBM](https://img.shields.io/badge/-LightGBM-02569B?style=flat-square)
![DVC](https://img.shields.io/badge/-DVC-945DD6?style=flat-square&logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/-MLflow%20(DagsHub)-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![AWS S3](https://img.shields.io/badge/-AWS%20S3-232F3E?style=flat-square&logo=amazonaws&logoColor=white)

</div>

<br/>

## 📌 Overview

This project trains a regression model that predicts food delivery time using restaurant and delivery-location coordinates, weather conditions, traffic density, and vehicle type. It's built as a **versioned, reproducible DVC pipeline** — not a notebook — with data versioning, experiment tracking, and a containerized Flask inference API, so the same pipeline that trains the model in dev can be reproduced and redeployed identically elsewhere.

<br/>

## 🧱 Pipeline Architecture

The DVC pipeline (`dvc.yaml`) runs as six sequential, dependency-tracked stages:

```
data_ingestion      → reads data/raw/data.csv, outputs cleaned data
data_preperation    → train/test split (params: test_size, random_state)
data_preprocessing  → preprocessing on train/test splits
train_model         → trains XGBoost + LightGBM, builds a stacking regressor
evaluate_model       → evaluates trained models on held-out test data
register_model       → registers the final model
```

Each stage only re-runs when its declared dependencies or params change (`dvc repro`), so the pipeline stays reproducible and fast to iterate on.

<br/>

## 📂 Project Structure

```
├── data/
│   ├── external/       # raw source CSV (DVC-tracked, not in git)
│   ├── raw/             # train/test split
│   ├── interim/         # cleaned data
│   └── processed/       # feature-engineered data
├── models/               # trained model artifacts (DVC-tracked)
├── src/
│   ├── data/             # ingestion + preprocessing
│   ├── features/         # feature engineering
│   └── models/           # training + evaluation
├── flask_app/            # inference API
├── notebooks/            # exploratory analysis
├── scripts/              # utility scripts
├── reports/figures/      # generated plots/metrics
├── .github/workflows/    # CI/CD
├── Dockerfile
├── dvc.yaml               # pipeline stages
├── params.yaml             # pipeline configuration
├── requirements.txt        # production deps (lean)
└── dev_requirements.txt    # full dev/CI toolchain
```

<br/>

## ⚙️ Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r dev_requirements.txt

dvc remote add -d s3remote s3://<your-bucket>/dvcstore
```

Place the raw dataset at `data/external/food_delivery.csv`, then version it with DVC:

```bash
dvc add data/external/food_delivery.csv
git add data/external/food_delivery.csv.dvc
git commit -m "Track raw dataset with DVC"
dvc push
```

<br/>

## ▶️ Run the Pipeline

```bash
dvc repro
```

This re-executes only the stages whose code, data, or params have changed, in dependency order — ingestion → preparation → preprocessing → training → evaluation → registration.

<br/>

## 🔐 Environment Variables (local + CI)

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 remote for DVC |
| `DAGSHUB_PAT` | Non-interactive MLflow experiment tracking auth (DagsHub) |

<br/>

## 🚀 Serve Locally

```bash
cd flask_app
python app.py
```

Or via Docker — the image installs `libgomp1` (required by LightGBM), copies the Flask app, the fitted preprocessor, and run metadata, then serves on port `8000`:

```bash
docker build -t swiggy-delivery-time .
docker run -p 8000:8000 swiggy-delivery-time
```

<br/>

## 🧠 Modeling

- **Gradient boosting**: XGBoost and LightGBM, tuned via `params.yaml`
- **Ensembling**: a stacking regressor combining both base learners
- **Preprocessing**: a fitted `power_transformer` for target/feature transformation, persisted as a pipeline artifact
- **Tracking**: experiments logged to MLflow via DagsHub for non-interactive CI tracking

<br/>

## 📄 License

No license specified yet — add one (e.g. MIT) if you intend this repo to be reused by others.

<br/>

<div align="center">

Built by [Sovith Kumar Singh](https://github.com/Sovith07) — [Portfolio](https://sovith07.github.io/) · [LinkedIn](https://www.linkedin.com/in/sovithkumarsingh)

</div>
