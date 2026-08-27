# Swiggy Delivery Time Prediction

End-to-end ML pipeline predicting food delivery time using restaurant/delivery
coordinates, weather, traffic, and vehicle features.

## Project structure

```
├── data/
│   ├── external/       # raw source CSV (DVC-tracked, not in git)
│   ├── raw/            # train/test split
│   ├── interim/        # cleaned data
│   └── processed/      # feature-engineered data
├── models/             # trained model artifacts (DVC-tracked)
├── src/
│   ├── data/           # ingestion + preprocessing
│   ├── features/       # feature engineering
│   └── models/         # training + evaluation
├── flask_app/          # inference API
├── dvc.yaml            # pipeline stages
├── params.yaml         # pipeline configuration
└── requirements.txt    # production deps (lean)
└── dev_requirements.txt # full dev/CI toolchain
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r dev_requirements.txt

dvc remote add -d s3remote s3://<your-bucket>/dvcstore
```

Place the raw dataset at `data/external/food_delivery.csv`, then:

```bash
dvc add data/external/food_delivery.csv
git add data/external/food_delivery.csv.dvc
git commit -m "Track raw dataset with DVC"
dvc push
```

## Run pipeline

```bash
dvc repro
```

## Environment variables (local + CI)

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 remote for DVC |
| `DAGSHUB_PAT` | Non-interactive MLflow tracking auth (DagsHub) |

## Serve locally

```bash
cd flask_app
python app.py
```
