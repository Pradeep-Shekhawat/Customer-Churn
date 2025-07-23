````markdown
# Telecom Customer Churn Risk Pipeline

An end‑to‑end Python project to ingest, clean, explore, model, and score telecom customer churn, with a companion Power BI dashboard for visualizing risk.

---

## Project Overview

Predict which telecom customers are most likely to churn by training a logistic regression and an XGBoost model on historical data. Store results in a lightweight pipeline, then expose the scores in Power BI for stakeholders to explore and act on.

---

## Features

- **Data Ingestion** from  CSV into a clean table  
- **Exploratory Data Analysis** in Jupyter notebook  
- **Feature Engineering**: ratio features, tenure buckets, one‑hot encoding  
- **Modeling**: balanced Logistic Regression and XGBoost classifier  
- **Evaluation**: classification reports & ROC AUC metrics  
- **Scoring**: reusable script to predict churn probabilities for new customers  
- **Automation**: single‑command orchestration via `run_all.py`

---

## Getting Started

### Prerequisites

- Python 3.8+  
- PostgreSQL (if using the DB ingestion path)  
- Power BI Desktop (for dashboard consumption)

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/customer-churn.git
   cd customer-churn
````

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   .\venv\Scripts\Activate.ps1   # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

Create a `.env` file in the project root (ignored by Git) with your database URL:

```dotenv
DATABASE_URL=postgresql://churn_user:YourSecretPassword@localhost:5432/churn_db
```

---

## Usage

### Run the Pipeline

Execute the full pipeline—from data prep through evaluation—by running:

```bash
python run_all.py
```

You’ll see training metrics printed (precision, recall, F1, ROC AUC) and models saved in `models/`.

### Generate & Score New Data

1. **Generate a sample** of 100 new customers (no churn label):

   ```bash
   python scripts/make_sample_new.py
   ```

   This writes `data/processed/new_customers.csv`.

2. **Score them**:

   ```bash
   python -m src.score --score
   ```

   Results land in `data/processed/churn_scores.csv`.

---

## Power BI Dashboard Overview

A companion Power BI dashboard connects to the scored data PostgreSQL and provides:

* **KPI Cards**: average churn probability & high‑risk count
* **Distribution Chart**: histogram of predicted churn scores
* **Segment Analysis**: avg. churn by contract, tenure bucket, service & payment
* **Top N Table**: highest‑risk customers
* **Filters & Drill‑through**: interactive slicers with drill‑through to customer details


---

## License

This project is released under the [MIT License]