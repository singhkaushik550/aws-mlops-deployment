import os
from io import StringIO

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ======================================
# CONFIGURATION
# ======================================

BUCKET = os.getenv(
    "S3_BUCKET",
    "mlops-house-predictions1"
)

AWS_REGION = os.getenv(
    "AWS_REGION",
    "ap-south-1"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING",
    "http://127.0.0.1:5000"
)

EXPERIMENT_NAME = "mlops-house-prediction"

MODEL_NAME = "house-price-predictor"


# ======================================
# AWS
# ======================================

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION
)


# ======================================
# FIND LATEST CLEANED DATA
# ======================================

def fetch_latest_data():

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix="processed/"
    )

    objects = response.get("Contents", [])

    csv_objects = [
        obj
        for obj in objects
        if obj["Key"].endswith(".csv")
    ]

    if not csv_objects:
        raise FileNotFoundError(
            "No cleaned CSV found in S3."
        )

    latest_object = max(
        csv_objects,
        key=lambda x: x["LastModified"]
    )

    key = latest_object["Key"]

    print(f"Reading: s3://{BUCKET}/{key}")

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    df = pd.read_csv(
        StringIO(
            obj["Body"].read().decode("utf-8")
        )
    )

    return df, key


# ======================================
# LOAD DATA
# ======================================

df, data_key = fetch_latest_data()

print(f"Fetched shape: {df.shape}")


# ======================================
# FEATURES / TARGET
# ======================================

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score",
]

TARGET = "price"


X = df[FEATURES]

y = df[TARGET]


# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ======================================
# MLFLOW
# ======================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ======================================
# TRAIN
# ======================================

with mlflow.start_run() as run:

    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )


    # ==================================
    # PREDICTION
    # ==================================

    predictions = model.predict(
        X_test
    )


    # ==================================
    # METRICS
    # ==================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    # ==================================
    # MLFLOW LOGGING
    # ==================================

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "data_source",
        f"s3://{BUCKET}/{data_key}"
    )

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "r2_score",
        r2
    )


    mlflow.sklearn.log_model(
    model,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )


    run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"


    print(
        f"\nMAE: {mae:.2f}"
        f"\nRMSE: {rmse:.2f}"
        f"\nR2: {r2:.4f}"
    )

    print(
        f"\nRun ID: {run_id}"
    )


# ======================================
# REGISTER MODEL
# ======================================

client = MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)

try:

    client.get_registered_model(
        MODEL_NAME
    )

except Exception:

    client.create_registered_model(
        MODEL_NAME
    )


model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)


print(
    f"Registered model version: "
    f"{model_version.version}"
)


# ======================================
# SET CHAMPION ALIAS
# ======================================

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    model_version.version
)


print(
    f"\nChampion model updated:"
    f" {MODEL_NAME}@champion"
)