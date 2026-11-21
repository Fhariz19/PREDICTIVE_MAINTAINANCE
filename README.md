# Predictive Maintenance System

## Project Overview

This project uses machine learning to predict potential machine failures based on machine operating conditions. It aims to support predictive maintenance by identifying possible failures before they occur.

## Dataset

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Source: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
- Total records: 10,000
- Target variable: Machine failure

## Machine Learning Model

- Selected model: Tuned Random Forest
- Decision threshold: 0.3959
- Number of input features: 10
- Model file: `models/final_predictive_maintenance_model.pkl`

## Final Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 98.80% |
| Precision | 80.00% |
| Recall | 86.27% |
| F1-score | 83.02% |
| ROC-AUC | 98.65% |
| PR-AUC | 90.30% |

## Installation

1. Clone this repository.
2. Open the project folder.
3. Install dependencies:

   `pip install -r requirements.txt`

## Run the Streamlit Application

`streamlit run app.py`

## Notes

- This project uses historical data from the AI4I 2020 dataset.
- The decision threshold is 0.3959.
- This is an academic prototype, not a replacement for professional machine inspection.

