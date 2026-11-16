
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "return_prediction_pipeline.joblib"

TARGET = "returned"

NUMERICAL_COLS = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "session_length_minutes",
    "num_product_views"
]

CATEGORICAL_COLS = [
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method"
]

BINARY_COLS = ["used_coupon"]

DROP_COLS = ["order_id", "delivery_delay_days"]


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

def clean_data(df):

    df = df.copy()

    # Remove invalid records, matching notebook rules
    df = df[
        (df["product_price"] >= 0) &
        (df["product_rating"].between(0, 5)) &
        (df["past_return_rate"].between(0, 1)) &
        (df["session_length_minutes"] >= 0) &
        (df["num_product_views"] >= 0)
    ].copy()

    # Impute negative discount values
    valid_discount_median = df.loc[
        df["discount_percent"] >= 0,
        "discount_percent"
    ].median()

    df.loc[
        df["discount_percent"] < 0,
        "discount_percent"
    ] = valid_discount_median

    print("Cleaned dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

def build_pipeline():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_COLS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ("binary", binary_pipeline, BINARY_COLS)
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model_pipeline


# --------------------------------------------------
# 5. TRAIN AND EVALUATE
# --------------------------------------------------

def train():

    df = load_data()
    df = clean_data(df)

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    print("\n--- Validation Results ---")

    print("Accuracy:", accuracy_score(y_val, predictions))
    print("Precision:", precision_score(y_val, predictions))
    print("Recall:", recall_score(y_val, predictions))
    print("F1 Score:", f1_score(y_val, predictions))
    print("ROC-AUC:", roc_auc_score(y_val, probabilities))

    print("\nClassification Report:")
    print(classification_report(y_val, predictions))

    # --------------------------------------------------
    # 6. TRAIN FINAL MODEL ON FULL CLEANED DATA
    # --------------------------------------------------

    final_pipeline = build_pipeline()

    final_pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, MODEL_PATH)

    print("\nFinal model saved successfully!")
    print("Model path:", MODEL_PATH)


if __name__ == "__main__":
    train()

