import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("PS2_Dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Target column
target_col = "Suggested Job Role"

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Numeric features
numeric_features = [
    "Logical quotient rating",
    "hackathons",
    "coding skills rating",
    "public speaking points"
]

# Categorical features
categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Smaller model
model = ExtraTreesClassifier(
    n_estimators=30,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Metadata for Streamlit
feature_meta = {}

for col in categorical_features:
    feature_meta[col] = sorted(
        df[col].dropna().unique().tolist()
    )

for col in numeric_features:
    feature_meta[col] = {
        "min": int(df[col].min()),
        "max": int(df[col].max()),
        "mean": float(df[col].mean())
    }

# Save model
model_data = {
    "pipeline": pipeline,
    "metadata": feature_meta,
    "target_classes": sorted(y.unique().tolist())
}

print("\nSaving compressed model...")

joblib.dump(
    model_data,
    "career_model_pipeline.pkl",
    compress=("xz", 9)
)

print("Model saved successfully!")
print("File: career_model_pipeline.pkl")
