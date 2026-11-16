
# E-commerce Order Return Prediction

## Project Overview

This project uses machine learning to predict whether an e-commerce order will be returned based on customer, product and shopping behaviour features.

## Problem Statement

Product returns create additional operational and logistical costs for e-commerce businesses. This project explores whether historical order data can help identify orders that are more likely to be returned.

## Dataset

- Training records: 200,000
- Features: 15 input columns
- Target: `returned`
- Cleaned training records: 147,434

## Models

The following classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression was selected as the final model.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.5678 |
| Precision | 0.5597 |
| Recall | 0.4408 |
| F1 Score | 0.4932 |
| ROC-AUC | 0.5893 |

## Project Structure

```text
ecommerce-return-prediction/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── scripts/
│   └── train.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd ecommerce-return-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python scripts/train.py
```

## Docker

Build the image:

```bash
docker build -t ecommerce-return-prediction .
```

Run the container:

```bash
docker run --rm ecommerce-return-prediction
```

## Limitations

The model achieved a modest ROC-AUC score, indicating limited predictive power from the available features. Further improvements may require better features and additional data.

## Author

Your Name
