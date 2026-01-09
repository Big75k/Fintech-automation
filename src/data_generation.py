import pandas as pd
import numpy as np
import os

def generate_credit_data(n_samples=5000):
    np.random.seed(42)
    user_ids = np.arange(1000, 1000 + n_samples)
    age = np.random.randint(18, 75, n_samples)
    income = np.random.normal(50000, 15000, n_samples).clip(15000, 200000)
    employment_years = np.random.randint(0, 40, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    debt_to_income = np.random.uniform(0.1, 0.6, n_samples)
    loan_amount = (income * np.random.uniform(0.1, 0.5, n_samples)).clip(1000, 50000)
    risk_score = (debt_to_income * 2) - (credit_score / 850) - (income / 200000) + np.random.normal(0, 0.1, n_samples)
    default = (risk_score > -0.5).astype(int)
    df = pd.DataFrame({
        'user_id': user_ids,
        'age': age,
        'income': income,
        'employment_years': employment_years,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'default': default
    })
    return df

def generate_churn_data(n_samples=5000):
    np.random.seed(43)
    user_ids = np.arange(1000, 1000 + n_samples)
    tenure_months = np.random.randint(1, 60, n_samples)
    transaction_count = np.random.randint(0, 100, n_samples)
    login_frequency = np.random.randint(1, 30, n_samples)
    support_tickets = np.random.randint(0, 10, n_samples)
    balance = np.random.normal(5000, 3000, n_samples).clip(0, 50000)
    churn_risk = (support_tickets * 0.2) - (login_frequency / 30) - (tenure_months / 60) + np.random.normal(0, 0.1, n_samples)
    churn = (churn_risk > -0.3).astype(int)
    df = pd.DataFrame({
        'user_id': user_ids,
        'tenure_months': tenure_months,
        'transaction_count': transaction_count,
        'login_frequency': login_frequency,
        'support_tickets': support_tickets,
        'balance': balance,
        'churn': churn
    })
    return df

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    credit_df = generate_credit_data(n_samples=500)
    credit_df.to_csv('data/raw/credit_data.csv', index=False)
    churn_df = generate_churn_data(n_samples=500)
    churn_df.to_csv('data/raw/churn_data.csv', index=False)
    print('Generated sample data in data/raw/')
