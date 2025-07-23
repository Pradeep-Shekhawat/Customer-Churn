import os
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def make_sample_new(db_uri: str,
                    output_csv: str = "data/processed/new_customers.csv",
                    n: int = 100,
                    random_state: int = 42):
    """
    1) Connects to PostgreSQL, reads telco_clean.
    2) Samples n rows, drops the churn column.
    3) Writes to data/processed/new_customers.csv.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load from DB
    engine = create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM telco_clean", engine)

    # Sample and drop the label
    new_df = df.sample(n=n, random_state=random_state).drop(columns=['churn'])

    # Write CSV
    new_df.to_csv(output_csv, index=False)
    print(f"Wrote sample new_customers.csv with {len(new_df)} rows to {output_csv}")

if __name__ == "__main__":
    make_sample_new(DB_URI)