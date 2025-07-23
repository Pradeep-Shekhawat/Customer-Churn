import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# locate .env (project root)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# now read it
DB_URI = os.getenv("DATABASE_URL")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take cleaned df, engineer features, and return a fully numeric DataFrame.
    """
    df = df.copy()

    # 1) Create ratio feature
    df['avg_monthly_charge'] = df['totalcharges'] / df['tenure'].replace(0, 1)

    # 2) Bucket tenure
    bins   = [0, 12, 24, df['tenure'].max()]
    labels = ['0-12', '13-24', '25+']
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # 3) Drop any remaining identifiers
    df = df.drop(columns=['customerid'], errors='ignore')

    # 4) One‑hot encode all object/category cols (excludes churn if still present)
    cats = df.select_dtypes(include=['object', 'category']).columns.difference(['churn'])
    df = pd.get_dummies(df, columns=cats, drop_first=True)

    return df

if __name__ == "__main__":
    import data.data_prep as dp
    df_clean = dp.load_and_clean(DB_URI)
    df_feat  = build_features(df_clean)
    print("Features shape:", df_feat.shape)