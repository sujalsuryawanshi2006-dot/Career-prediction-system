import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # 1. Load the dataset
    dataset_path = "PS2_Dataset.csv"
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Clean up column names (strip trailing spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Separate features and target
    target_col = 'Suggested Job Role'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Identify numeric and categorical features
    numeric_features = [
        'Logical quotient rating',
        'hackathons',
        'coding skills rating',
        'public speaking points'
    ]
    
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    
    # 4. Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # 5. Define Model & Pipeline
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 7. Train the Model
    print("Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    # 8. Evaluate the Model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save feature names and categories to log or save inside model for streamlit
    # Let's save the list of categories for each categorical feature so Streamlit can build dropdowns
    feature_meta = {}
    for col in categorical_features:
        # Sort and store unique categories
        feature_meta[col] = sorted(list(df[col].dropna().unique()))
        
    for col in numeric_features:
        feature_meta[col] = {
            'min': int(df[col].min()),
            'max': int(df[col].max()),
            'mean': float(df[col].mean())
        }
    
    # Save the pipeline and the metadata
    model_data = {
        'pipeline': pipeline,
        'metadata': feature_meta,
        'target_classes': sorted(list(y.unique()))
    }
    
    model_filename = 'career_model_pipeline.pkl'
    print(f"Saving model and metadata to {model_filename}...")
    joblib.dump(model_data, model_filename)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
