from src.data.data_prep         import load_and_clean
from src.features.build_features import build_features
from src.model.train           import train_models
from src.model.evaluate        import evaluate_models
from src.score                  import score_new
from sklearn.model_selection    import train_test_split
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def main():
    # 1) Load & clean
    df = load_and_clean(DB_URI)

    # 2) Extract target & drop id
    y  = df['churn'].map({'No': 0, 'Yes': 1})
    df = df.drop(columns=['churn', 'customerid'])

    # 3) Feature engineering
    X = build_features(df)

    # 4) Train models
    models, splits = train_models(X, y, output_dir="models")
    X_train, X_test, y_train, y_test = splits

    # 5) Evaluate
    model_paths = {
        'logistic': "models/logistic_model.joblib",
        'xgboost':  "models/xgb_model.joblib"
    }
    evaluate_models(X_test, y_test, model_paths)

    # 6) Score new data (if exists)
    try:
        score_new(
            model_path="models/xgb_model.joblib",
            input_csv="data/processed/new_customers.csv",
            output_csv="data/processed/churn_scores.csv"
        )
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()