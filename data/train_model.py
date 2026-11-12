import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# 1. Load dataset
data = pd.read_csv("data/claims.csv")

print("Dataset loaded successfully!")
print("Original shape:", data.shape)

# 2. Data preprocessing
data.drop_duplicates(inplace=True)

# Remove records with missing target values
data.dropna(subset=["status"], inplace=True)

# Remove ID because it is only an identifier
X = data.drop(columns=["id", "status"])
y = data["status"]

# 3. Feature engineering
X["amount_per_age"] = X["billed_amount"] / X["patient_age"]

# Identify numerical and categorical features
numeric_features = [
    "billed_amount",
    "patient_age",
    "amount_per_age"
]

categorical_features = [
    "payer",
    "provider",
    "cpt_code",
    "diagnosis_code"
]

# 4. Preprocessing pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

# 6. Build complete ML pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Model evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["Paid", "Partially_Paid", "Denied"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["Paid", "Partially_Paid", "Denied"]
))

# 10. Save complete pipeline
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/claim_prediction_pipeline.pkl")

print("\nComplete ML pipeline saved successfully!")
print("Location: models/claim_prediction_pipeline.pkl")

