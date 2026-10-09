# credit-card-fraud-detection
Credit card fraud detection using machine learning, imbalanced data classification, and anomaly detection.
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# 1. Generate an imbalanced dataset for demonstration
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    weights=[0.99, 0.01],
    random_state=42
)

X = pd.DataFrame(X)
y = pd.Series(y)

print("CREDIT CARD FRAUD DETECTION")
print("-" * 40)
print("Total transactions:", len(y))
print("Normal transactions:", (y == 0).sum())
print("Fraudulent transactions:", (y == 1).sum())
print(f"Fraud percentage: {y.mean() * 100:.2f}%")

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Evaluation function
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Fraud"],
            zero_division=0
        )
    )

    print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
    print(
        "Average Precision:",
        round(average_precision_score(y_test, scores), 4)
    )


# 4. Logistic Regression
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )
)

logistic_model.fit(X_train, y_train)

evaluate_model(
    "LOGISTIC REGRESSION",
    logistic_model,
    X_test,
    y_test
)


# 5. Random Forest
forest_model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

forest_model.fit(X_train, y_train)

evaluate_model(
    "RANDOM FOREST",
    forest_model,
    X_test,
    y_test
)


# 6. Isolation Forest anomaly detection
# Train using only normal transactions
normal_data = X_train[y_train == 0]

anomaly_model = make_pipeline(
    StandardScaler(),
    IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
)

anomaly_model.fit(normal_data)

# Isolation Forest: -1 = anomaly, 1 = normal
raw_predictions = anomaly_model.predict(X_test)
anomaly_predictions = (raw_predictions == -1).astype(int)

# Lower decision scores mean more anomalous
anomaly_scores = -anomaly_model.decision_function(X_test)

print("\n" + "=" * 45)
print("ISOLATION FOREST")
print("=" * 45)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, anomaly_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        anomaly_predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, anomaly_scores), 4)
)

print(
    "Average Precision:",
    round(average_precision_score(y_test, anomaly_scores), 4)
)

print("\nFraud detection analysis completed!")

