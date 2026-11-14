

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# Remove rows without target values
df = df.dropna(subset=["SalePrice"])

# Separate features and target
X = df.drop(
    columns=["SalePrice", "Order", "PID"],
    errors="ignore"
)

y = df["SalePrice"]

# Handle structural missing values
categorical_none_columns = [
    "Alley", "Mas Vnr Type", "Bsmt Qual", "Bsmt Cond",
    "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond", "Pool QC",
    "Fence", "Misc Feature"
]

numeric_zero_columns = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Total Bsmt SF", "Bsmt Full Bath",
    "Bsmt Half Bath", "Garage Yr Blt", "Garage Cars",
    "Garage Area"
]

for col in categorical_none_columns:
    if col in X.columns:
        X[col] = X[col].fillna("None")

for col in numeric_zero_columns:
    if col in X.columns:
        X[col] = X[col].fillna(0)


# ==========================================
# CREATE CLEANED CSV - TRAINING STATISTICS ONLY
# ==========================================

# Split indices using the same random state as the model
train_indices, test_indices = train_test_split(
    X.index,
    test_size=0.20,
    random_state=42
)

# Create a copy for the cleaned CSV
cleaned_df = X.copy()

# Identify numerical and categorical columns
numeric_columns = cleaned_df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = cleaned_df.select_dtypes(
    include=["object"]
).columns

# Calculate numerical medians using training rows only
train_medians = cleaned_df.loc[
    train_indices, numeric_columns
].median()

# Calculate categorical modes using training rows only
train_modes = {}

for col in categorical_columns:
    mode_values = cleaned_df.loc[
        train_indices, col
    ].mode()

    if not mode_values.empty:
        train_modes[col] = mode_values[0]

# Apply training-set statistics to both training and test rows
cleaned_df[numeric_columns] = cleaned_df[
    numeric_columns
].fillna(train_medians)

cleaned_df[categorical_columns] = cleaned_df[
    categorical_columns
].fillna(train_modes)

# Apply One-Hot Encoding
cleaned_df = pd.get_dummies(
    cleaned_df,
    columns=categorical_columns,
    dtype=int
)

# Add target variable
cleaned_df["SalePrice"] = y

# Save cleaned dataset
cleaned_df.to_csv(
    "cleaned_ames_housing.csv",
    index=False
)

print("Cleaned CSV saved successfully!")
print("Cleaned dataset shape:", cleaned_df.shape)
print(
    "Remaining missing values:",
    cleaned_df.isnull().sum().sum()
)


# ==========================================
# BASELINE REGRESSION MODEL
# ==========================================

# Split data for model training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Identify feature types
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Numerical preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nUrbanNest Baseline Regression Results")
print("-------------------------------------")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

