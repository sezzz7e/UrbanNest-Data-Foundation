
# UrbanNest - Data Foundation and Baseline Regression

## Project Overview
This project prepares the Ames Housing dataset for machine learning
and builds a baseline Linear Regression model to predict house prices.

## Dataset Source
Kaggle: https://www.kaggle.com/datasets/preathm/ames-housing-dataset

## Dataset Information
- Original records: 2930
- Features used: 79
- Target variable: SalePrice

## Data Cleaning and Preprocessing

The cleaned dataset is generated directly by `baseline_model.py` and saved as `cleaned_ames_housing.csv`.

The preprocessing steps include:

- Replacing structural missing categorical values with `None`.
- Replacing structural missing numerical values with `0`.
- Handling remaining numerical missing values using median imputation.
- Handling remaining categorical missing values using mode imputation.
- Applying One-Hot Encoding to categorical features.
- Saving the processed dataset as a CSV file with no missing values.

The baseline model uses a separate preprocessing pipeline fitted on the training data for model training and evaluation.
## Model
Linear Regression

## Evaluation Results
- MAE: $17,309.70
- RMSE: $29,633.80
- R2 Score: 0.8905

## Project Files
- cleaned_ames_housing.csv: Cleaned dataset.
- baseline_model.py: Standalone regression script.
- baseline_results.json: Evaluation results.
- baseline_results.txt: Readable evaluation summary.
- feature_importance.csv: Feature importance results.

## Tools and Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.

## How to Run
1. Install dependencies using requirements.txt.
2. Place AmesHousing.csv in the project directory.
3. Run: python baseline_model.py
