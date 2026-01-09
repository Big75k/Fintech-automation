import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF

def generate_visualizations():
    os.makedirs('reports/figures', exist_ok=True)
    credit_df = pd.read_csv('data/raw/credit_data.csv')
    plt.figure(figsize=(10, 6))
    sns.histplot(data=credit_df, x='credit_score', hue='default', multiple='stack')
    plt.title('Credit Score Distribution by Default Status')
    plt.savefig('reports/figures/credit_score_dist.png')
    plt.close()
    churn_df = pd.read_csv('data/raw/churn_data.csv')
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=churn_df, x='churn', y='support_tickets')
    plt.title('Support Tickets vs Churn')
    plt.savefig('reports/figures/churn_support_tickets.png')
    plt.close()
    print('Visualizations saved to reports/figures/')

def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Fintech Data Science Automation Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="This report summarizes the results of the automated credit risk and customer churn prediction pipelines.")
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="1. Credit Risk Assessment", ln=True)
    pdf.image('reports/figures/credit_score_dist.png', x=10, y=None, w=180)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="2. Customer Churn Prediction", ln=True)
    pdf.image('reports/figures/churn_support_tickets.png', x=10, y=None, w=180)
    os.makedirs('reports', exist_ok=True)
    pdf.output("reports/fintech_analysis_report.pdf")
    print('PDF report generated: reports/fintech_analysis_report.pdf')

if __name__ == '__main__':
    generate_visualizations()
    create_pdf_report()
