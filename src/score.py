import os
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from src.features.build_features import build_features
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def generate_sample(db_uri: str,
                    output_csv: str = "data/processed/new_customers.csv",
                    n: int = 100,
                    random_state: int = 42):
    """
    Create a sample 'new_customers.csv' by pulling n rows
    from telco_clean, dropping the churn label.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    engine = create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM telco_clean", engine)
    new_df = df.sample(n=n, random_state=random_state).drop(columns=['churn'])
    new_df.to_csv(output_csv, index=False)
    print(f"Wrote sample new_customers.csv ({len(new_df)} rows)")

def score_new(model_path: str,
              input_csv: str  = "data/processed/new_customers.csv",
              output_csv: str = "data/processed/churn_scores.csv"):
    """
    Read new_customers.csv, build features, load model,
    predict churn_probability, and write churn_scores.csv.
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"{input_csv} not found — run generate_sample first?")
    df_new = pd.read_csv(input_csv)
    X_new  = build_features(df_new)
    model = joblib.load(model_path)
    df_new['churn_probability'] = model.predict_proba(X_new)[:, 1]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_new[['customerid','churn_probability']].to_csv(output_csv, index=False)
    print(f"Wrote churn_scores.csv ({len(df_new)} rows)")


if __name__ == "__main__":
    import argparse

    MODEL_PATH  = "models/xgb_model.joblib"
    parser = argparse.ArgumentParser(
        description="Generate sample, score, and load for dashboard"
    )
    parser.add_argument("--sample", action="store_true",
                        help="Generate data/processed/new_customers.csv")
    parser.add_argument("--score",  action="store_true",
                        help="Score new_customers.csv → churn_scores.csv")
    parser.add_argument("--to-db",  action="store_true",
                        help="Load churn_scores.csv into a SQL table")
    args = parser.parse_args()

    if args.sample:
        generate_sample(DB_URI)
    if args.score:
        score_new(MODEL_PATH)