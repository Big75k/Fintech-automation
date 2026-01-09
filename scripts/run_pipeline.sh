#!/usr/bin/env bash
set -euo pipefail

# Generate sample data and run training + reporting
python -m src.data_generation
python -m src.credit_risk
python -m src.churn_prediction
python -m src.report_gen || true

echo "Pipeline finished. Models are in models/ and reports/."
