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

def load_and_clean(db_uri: str) -> pd.DataFrame:
    """
    Connect to PostgreSQL, load telco_clean, handle nulls in totalcharges,
    and return the cleaned DataFrame.
    """
    engine = create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM telco_clean", engine)

    # Impute totalcharges nulls as monthlycharges * tenure
    df['totalcharges'] = df['totalcharges'].fillna(
        df['monthlycharges'] * df['tenure']
    )
    return df

if __name__ == "__main__":
    df = load_and_clean(DB_URI)
    print(f"Loaded {len(df)} rows; nulls remaining: {df.isna().sum().sum()}")