
from src.data_generation import generate_credit_data, generate_churn_data

def test_generate_credit_data():
    df = generate_credit_data(n_samples=10)
    assert not df.empty
    assert 'default' in df.columns

def test_generate_churn_data():
    df = generate_churn_data(n_samples=10)
    assert not df.empty
    assert 'churn' in df.columns
