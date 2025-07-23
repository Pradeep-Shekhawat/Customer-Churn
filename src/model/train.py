import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def train_models(X, y, output_dir: str = "models") -> dict:
    """
    Split X/y, train LogisticRegression and XGBoost, save models, and return them.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {}

    # 1. Logistic Regression pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, os.path.join(output_dir, "logistic_model.joblib"))
    models['logistic'] = pipe

    # 2. XGBoost classifier
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale,
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, os.path.join(output_dir, "xgb_model.joblib"))
    models['xgboost'] = xgb_clf

    return models, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    import pandas as pd
    from data.data_prep       import load_and_clean
    from features.build_features import build_features

    df = load_and_clean(DB_URI)

    # Extract target
    y  = df['churn'].map({'No': 0, 'Yes': 1})
    df = df.drop(columns=['churn', 'customerid'])

    # Features
    X = build_features(df)

    models, splits = train_models(X, y, output_dir="../models")
    print("Trained models:", list(models.keys()))