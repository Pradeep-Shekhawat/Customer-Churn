import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def load_scores(db_uri: str,
                scores_csv: str = "data/processed/churn_scores.csv",
                table_name: str = "churn_scores"):
    """
    1) Reads churn_scores.csv.
    2) Creates (or replaces) a PostgreSQL table named churn_scores.
    3) Writes the DataFrame back to the DB, so Power BI can connect directly.
    """
    df = pd.read_csv(scores_csv)

    engine = create_engine(db_uri)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
        df.to_sql(table_name, conn, index=False)
    print(f"Loaded {len(df)} rows into DB table {table_name}")

if __name__ == "__main__":
    load_scores(DB_URI)