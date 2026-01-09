import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_credit_model(data_path='data/raw/credit_data.csv', output_dir='models'):
    df = pd.read_csv(data_path)
    X = df.drop(['user_id', 'default'], axis=1)
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print('Credit Risk Model Performance:')
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'credit_risk_model.pkl'))
    print(f'Model saved to {os.path.join(output_dir, "credit_risk_model.pkl")}')
    return model

if __name__ == '__main__':
    train_credit_model()
