import joblib
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def evaluate_models(X_test, y_test, model_paths: dict):
    """
    Load models from paths, evaluate on X_test/y_test, and print metrics.
    """
    for name, path in model_paths.items():
        model = joblib.load(path)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        print(f"\n=== {name.upper()} ===")
        print(classification_report(y_test, preds))
        print("ROC AUC:", roc_auc_score(y_test, probs))

if __name__ == "__main__":
    from train import train_models
    from data.data_prep import load_and_clean
    from features.build_features import build_features
    from sklearn.model_selection import train_test_split

    df = load_and_clean(DB_URI)
    y  = df['churn'].map({'No': 0, 'Yes': 1})
    df = df.drop(columns=['churn', 'customerid'])
    X = build_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_paths = {
        'logistic': "../models/logistic_model.joblib",
        'xgboost':  "../models/xgb_model.joblib"
    }
    evaluate_models(X_test, y_test, model_paths)