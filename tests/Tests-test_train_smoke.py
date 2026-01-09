import os
from src.data_generation import generate_credit_data, generate_churn_data
from src.credit_risk import train_credit_model
from src.churn_prediction import train_churn_model

def test_training_smoke(tmp_path):
    d = tmp_path / 'data' / 'raw'
    d.mkdir(parents=True)
    credit = generate_credit_data(n_samples=100)
    churn = generate_churn_data(n_samples=100)
    credit.to_csv(d / 'credit_data.csv', index=False)
    churn.to_csv(d / 'churn_data.csv', index=False)
    # Run training; models will be written to tmp_path / 'models'
    train_credit_model(data_path=str(d / 'credit_data.csv'), output_dir=str(tmp_path / 'models'))
    train_churn_model(data_path=str(d / 'churn_data.csv'), output_dir=str(tmp_path / 'models'))
    assert (tmp_path / 'models' / 'credit_risk_model.pkl').exists()
    assert (tmp_path / 'models' / 'churn_model.pkl').exists()
