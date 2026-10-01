import os
from datetime import date

import boto3
import pandas as pd


# =========================
# CONFIGURATION
# =========================

LOCAL_DATA_PATH = os.getenv(
    "LOCAL_DATA_PATH",
    "/Users/kaushik/DataSet/Mlops_house_predication_raw_data.csv"
)

BUCKET = os.getenv(
    "S3_BUCKET",
    "mlops-house-predictions1"
)

AWS_REGION = os.getenv(
    "AWS_REGION",
    "ap-south-1"
)


# =========================
# LOAD DATA
# =========================

print("Loading data...")

df = pd.read_csv(LOCAL_DATA_PATH)

print("\n=========== Before Cleaning ============")
print(df.isnull().sum())
print(f"Shape Before: {df.shape}")


# =========================
# CLEAN DATA
# =========================

df_clean = df.dropna()

print("\n=========== After Cleaning ============")
print(df_clean.isnull().sum())
print(f"Shape After: {df_clean.shape}")


# =========================
# SAVE CLEAN DATA LOCALLY
# =========================

clean_path = "/Users/kaushik/DataSet/Mlops_house_predication_clean.csv"

df_clean.to_csv(
    clean_path,
    index=False
)

print(f"\nCleaned data saved locally: {clean_path}")


# =========================
# UPLOAD TO S3
# =========================

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION
)

today = date.today().isoformat()

s3_key = (
    f"processed/{today}/"
    f"Mlops_house_predication_clean.csv"
)

print("\nUploading cleaned data to S3...")

s3.upload_file(
    clean_path,
    BUCKET,
    s3_key
)

print(
    f"\nUploaded successfully:"
    f"\ns3://{BUCKET}/{s3_key}"
)