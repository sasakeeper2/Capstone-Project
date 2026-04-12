# Databricks notebook source
# MAGIC %md
# MAGIC # Stock + Reddit Multi-Stock ML Pipeline
# MAGIC
# MAGIC This notebook:
# MAGIC - Downloads **multi-ticker stock data** automatically
# MAGIC - Reuses the same Reddit text per day
# MAGIC - Adds **ticker awareness** so all stocks contribute
# MAGIC - Uses **time-based train/test split**
# MAGIC - Includes diagnostics to verify ticker balance
# MAGIC - Logs Experiments to mlflow
# MAGIC - Is now connected to GitHub
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 2: Install Dependencies
# Step 1: Install required packages for workflow
!pip install yfinance openpyxl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download Stock Data

# COMMAND ----------

# DBTITLE 1,Cell 4: Download Multi-Ticker Stock Data
# Step 2: Download and process daily stock price/volume for multiple tickers
import yfinance as yf
import pandas as pd

START = "2025-10-01"
END = "2026-02-27"

tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "BAC", "SPY", "GOOGL", "GOOG", "BRK-B", "UNH","V", "MA", "HD", "DIS", "PEP", "KO", "NFLX", "ADBE", "CRM", "INTC", "CSCO", "WMT", "PG", "XOM", "CVX", "LLY", "MRK", "ABBV", "T", "VZ", "ORCL", "IBM", "MCD", "NKE", "COST", "TMO", "ABT"
]

all_data = []
for ticker in tickers:
    df_stock = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
        progress=False
    ).reset_index()
    # Handle MultiIndex columns if present
    if isinstance(df_stock.columns, pd.MultiIndex):
        df_stock.columns = [col[0] for col in df_stock.columns]
    df_stock["ticker"] = ticker
    df_stock["date"] = pd.to_datetime(df_stock["Date"]).dt.date
    df_stock = df_stock[["date", "ticker", "Close", "Volume"]]
    all_data.append(df_stock)
# Combine all tickers
numerical_data = pd.concat(all_data, ignore_index=True)
numerical_data.to_csv("numerical_data.csv", index=False)
print("Saved numerical_data.csv with shape:", numerical_data.shape)
numerical_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Stock Data

# COMMAND ----------

# DBTITLE 1,Cell 6: Load Stock Data and Setup
# Step 3: Load stock price/volume data and initialize pipeline imports
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

df = pd.read_csv("numerical_data.csv")
df["date"] = pd.to_datetime(df["date"]).dt.date
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Stock Feature Engineering (Per Ticker)

# COMMAND ----------

# DBTITLE 1,Add Day-Of-Week Feature
# Step 4: Compute returns, volatility, price features + add day of week
for period in [1, 3, 5]:
    df[f"ret_{period}"] = df.groupby("ticker")["Close"].pct_change(period)

# Calculate rolling volatility
df["vol_10"] = (
    df.groupby("ticker")["ret_1"]
    .rolling(10)
    .std()
    .reset_index(0, drop=True)
)

# Calculate rolling SMA and ratio to close
sma_10 = (
    df.groupby("ticker")["Close"]
    .rolling(10)
    .mean()
    .reset_index(0, drop=True)
)
df["sma_ratio"] = df["Close"] / sma_10

# Add day-of-week as numeric feature (0=Monday, 6=Sunday)
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Prediction Targets

# COMMAND ----------

# DBTITLE 1,Cell 10: Create Prediction Targets
# Step 5: Create prediction targets for 1d, 3d, 5d closes (future values)
df["target_1d"] = df.groupby("ticker")["Close"].shift(-1)
df["target_3d"] = df.groupby("ticker")["Close"].shift(-3)
df["target_5d"] = df.groupby("ticker")["Close"].shift(-5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load & Aggregate Reddit Text

# COMMAND ----------

# DBTITLE 1,Cell 12: Load & Aggregate Reddit Text
# Step 6: Load and merge daily Reddit text posts for each date
# Clean text field import and combine posts

# Load Reddit text data from Excel

# Group by date: aggregate all text into one string per day

df_reddit = pd.read_excel("/Workspace/Users/sasakeeper2@gmail.com/Capstone/(Clone) combined_text_by_date.xlsx")
df_reddit["date"] = pd.to_datetime(df_reddit["date"]).dt.date

reddit_daily = (
    df_reddit.groupby("date")["text"]
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)

reddit_daily.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Merge Stock + Reddit Data

# COMMAND ----------

# DBTITLE 1,Cell 14: Merge Stock + Reddit Data
# Step 7: Merge Reddit text to stock data by date
# Fill missing text with empty string

df = df.merge(reddit_daily, on="date", how="left")
df["text"] = df["text"].fillna("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Clean Text

# COMMAND ----------

# DBTITLE 1,Cell 16: Clean Text
# Step 8: Clean Reddit text by removing URLs, non-alpha, lowercase

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Apply text cleaning

df["clean_text"] = df["text"].apply(clean_text)

# COMMAND ----------

# DBTITLE 1,Cell 17: Install TextBlob for Sentiment
# Step 9: Install TextBlob to prepare for sentiment feature extraction
%pip install textblob

# COMMAND ----------

# DBTITLE 1,Cell 18: Compute Text Sentiment Polarity
# Step 10: Compute sentiment score for each sample using TextBlob
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Add sentiment polarity feature
df["sentiment"] = df["clean_text"].apply(get_sentiment)

# COMMAND ----------

# DBTITLE 1,Cell 19: Engineer Lagged/Rolling/Interaction Features
# Step 11: Engineer rolling averages, lagged sentiment, and feature interactions
for window in [3, 7, 14]:
    df[f'sentiment_mean_{window}d'] = df.groupby('ticker')['sentiment'].rolling(window).mean().reset_index(0, drop=True)
    df[f'sentiment_std_{window}d'] = df.groupby('ticker')['sentiment'].rolling(window).std().reset_index(0, drop=True)
    df[f'Close_mean_{window}d'] = df.groupby('ticker')['Close'].rolling(window).mean().reset_index(0, drop=True)
    df[f'Close_std_{window}d'] = df.groupby('ticker')['Close'].rolling(window).std().reset_index(0, drop=True)
# Feature interactions
for feat in ['vol_10', 'ret_1', 'sma_ratio']:
    df[f'sentiment_x_{feat}'] = df['sentiment'] * df[feat]
print("Engineered feature columns:", [col for col in df.columns if 'sentiment_' in col or 'Close_' in col or 'sentiment_x_' in col])

# COMMAND ----------

# DBTITLE 1,Cell 20: Final Modeling Dataset (with engineered features)
# Step 12: Create modeling DataFrame with all features (including engineered)
features = ["ret_1", "ret_3", "ret_5", "vol_10", "sma_ratio", "sentiment", "day_of_week"]
eng_feats = [col for col in df.columns if (
    (col.startswith('sentiment_') or col.startswith('Close_mean_') or col.startswith('Close_std_') or col.startswith('sentiment_x_'))
    and col not in features)]
all_features = features + eng_feats

target = "target_1d"

df_model = df.dropna(subset=all_features + [target]).reset_index(drop=True)
df_model = df_model[df_model["text"].str.strip() != ""]
df_model["ticker_id"] = (
    df_model["ticker"]
    .astype("category")
    .cat.codes
)
df_model.head()
df_model.to_csv("df_model.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Diagnostics: Verify All Tickers Are Used

# COMMAND ----------

# DBTITLE 1,Cell 22: Ticker/Date Diagnostics
# Step 13: Print diagnostics for ticker/row counts and date coverage
print("Rows per ticker:")
print(df_model["ticker"].value_counts())
print("\nDate range:")
print(df_model["date"].min(), "->", df_model["date"].max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Time-Based Train/Test Split

# COMMAND ----------

# DBTITLE 1,Cell 24: Time-Based Train/Test Split
# Step 14: Split data into train/test by date for realistic evaluation
split_date = df_model["date"].quantile(0.8)

train_df = df_model[df_model["date"] <= split_date]
test_df  = df_model[df_model["date"] > split_date]

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)
train_df.head()

# COMMAND ----------

# DBTITLE 1,Step 14.1: Add Validation Set Split (time-based)
# Step 14.1: Split data into train/validation/test by date for robust evaluation
train_quantile = 0.7
val_quantile = 0.85

train_end_date = df_model["date"].quantile(train_quantile)
val_end_date   = df_model["date"].quantile(val_quantile)

train_df = df_model[df_model["date"] <= train_end_date]
val_df   = df_model[(df_model["date"] > train_end_date) & (df_model["date"] <= val_end_date)]
test_df  = df_model[df_model["date"] > val_end_date]

print("Train size:", train_df.shape)
print("Validation size:", val_df.shape)
print("Test size:", test_df.shape)
print("Validation split dates:", train_end_date, val_end_date)
train_df.head()

# COMMAND ----------

# DBTITLE 1,Step 14.2: Update Feature Scaling for Validation
# Step 14.2: Scale expanded features for train, validation, and test sets
additional_feats = [
    col for col in train_df.columns
    if (
        ('sentiment_' in col or 'Close_mean_' in col or 'Close_std_' in col or 'sentiment_x_' in col)
        and col not in features
    )
]
num_features = ["ticker_id"] + features + additional_feats

scaler = StandardScaler()
X_num_train = scaler.fit_transform(train_df[num_features])
X_num_val   = scaler.transform(val_df[num_features])
X_num_test  = scaler.transform(test_df[num_features])

X_train = X_num_train
X_val   = X_num_val
X_test  = X_num_test

y_train = train_df[target].values
y_val   = val_df[target].values
y_test  = test_df[target].values

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)
import pandas as pd
display(pd.DataFrame(X_train, columns=num_features).head())
display(pd.DataFrame(X_val, columns=num_features).head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Scale Numeric Features (Train Only)

# COMMAND ----------

# DBTITLE 0,Cell 26: Expanded Feature Set for Modeling
# Step 15: Scale expanded numeric & engineered feature set for model input
additional_feats = [
    col for col in train_df.columns 
    if (
        ('sentiment_' in col or 'Close_mean_' in col or 'Close_std_' in col or 'sentiment_x_' in col) 
        and col not in features
    )
]
num_features = ["ticker_id"] + features + additional_feats
scaler = StandardScaler()
X_num_train = scaler.fit_transform(train_df[num_features])
X_num_test  = scaler.transform(test_df[num_features])
X_train = X_num_train
X_test  = X_num_test
y_train = train_df[target].values
y_test = test_df[target].values
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
import pandas as pd
display(pd.DataFrame(X_train, columns=num_features).head())
pd.DataFrame(X_train, columns=num_features).to_csv("X_train.csv")

# COMMAND ----------

# DBTITLE 1,Step 16: Visualize Sentiment Distribution
# Step 16: Visualize the distribution of daily sentiment scores
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(df['sentiment'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Step 17: Engineer Lagged/Rolling/Interaction Features
# Step 17: Add rolling averages, standard deviations, and interaction features (repeated for completeness)
for window in [3, 7, 14]:
    df[f'sentiment_mean_{window}d'] = df.groupby('ticker')['sentiment'].rolling(window).mean().reset_index(0, drop=True)
    df[f'sentiment_std_{window}d'] = df.groupby('ticker')['sentiment'].rolling(window).std().reset_index(0, drop=True)
    df[f'Close_mean_{window}d'] = df.groupby('ticker')['Close'].rolling(window).mean().reset_index(0, drop=True)
    df[f'Close_std_{window}d'] = df.groupby('ticker')['Close'].rolling(window).std().reset_index(0, drop=True)

# Interaction features
for feat in ['vol_10', 'ret_1', 'sma_ratio']:
    df[f'sentiment_x_{feat}'] = df['sentiment'] * df[feat]
print("Engineered feature columns:", [col for col in df.columns if 'sentiment_' in col or 'Close_' in col or 'sentiment_x_' in col])

# COMMAND ----------

# DBTITLE 1,Step 18: Correlation Analysis of Features
# Step 18: Analyze correlations of numeric (including engineered) features
import seaborn as sns
feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int'] and col != 'target_5d']
correlation_matrix = df[feature_cols + ['target_1d']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Feature Correlation Matrix (incl. engineered features)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Step 19: Display First Rows of Model Input Data
# Step 19: Display first five rows of scaled model input data and feature columns
import pandas as pd
display(pd.DataFrame(X_train, columns=num_features).head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Train & Evaluate Model

# COMMAND ----------

# DBTITLE 1,Initialize MLflow Experiment Tracking
# Initialize MLflow experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.keras

# Set experiment name
mlflow.set_experiment("/Users/sasakeeper2@gmail.com/stock-reddit-ml-pipeline")

print("MLflow experiment initialized: stock-reddit-ml-pipeline")

# COMMAND ----------

# MAGIC %md
# MAGIC # Ridge Regression

# COMMAND ----------

# DBTITLE 1,Step 20: Ridge Regression (Hyperparameter Tuning + Evaluation)
# Step 20: Fit Ridge Regression with grid search on alpha + MLflow tracking
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="Ridge_Regression"):
    # Log parameter grid
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    mlflow.log_params({"param_grid": str(param_grid)})
    
    # Grid search with 5-fold CV
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring="neg_root_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)
    ridge_model = grid.best_estimator_
    
    # Log ALL tested parameter combinations
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(run_name=f"Ridge_combo_{i}", nested=True):
            for param, value in params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("cv_mean_score", -grid.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std_score", grid.cv_results_['std_test_score'][i])
    
    # Log best hyperparameters
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # Make predictions
    train_preds_ridge = ridge_model.predict(X_train)
    val_preds_ridge   = ridge_model.predict(X_val)
    test_preds_ridge  = ridge_model.predict(X_test)
    
    # Calculate metrics
    train_rmse_ridge = np.sqrt(mean_squared_error(y_train, train_preds_ridge))
    val_rmse_ridge   = np.sqrt(mean_squared_error(y_val, val_preds_ridge))
    test_rmse_ridge  = np.sqrt(mean_squared_error(y_test, test_preds_ridge))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_ridge)
    mlflow.log_metric("val_rmse", val_rmse_ridge)
    mlflow.log_metric("test_rmse", test_rmse_ridge)
    mlflow.log_metric("cv_best_score", -grid.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(ridge_model, "ridge_model")
    
    print("Best alpha:", grid.best_params_["alpha"])
    print("Train Ridge RMSE:", train_rmse_ridge)
    print("Validation Ridge RMSE:", val_rmse_ridge)
    print("Test Ridge RMSE:", test_rmse_ridge)

# COMMAND ----------

# DBTITLE 1,Ridge Model Save & Confusion Matrix
# Save Ridge model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
joblib.dump(ridge_model, 'ridge_best.pkl')
print('Ridge model saved as ridge_best.pkl')

# Bin predictions and true values for confusion matrix
bins = 5
val_bins_pred = pd.cut(val_preds_ridge, bins, labels=False)
val_bins_true = pd.cut(y_val, bins, labels=False)
cm = confusion_matrix(val_bins_true, val_bins_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Bin')
plt.ylabel('Actual Bin')
plt.title('Ridge Regression Validation Confusion Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Support Vectors

# COMMAND ----------

# DBTITLE 1,Step 21: Support Vector Regression (Hyperparameter Tuning + Evaluation)
# Step 21: Fit SVR model with grid search + MLflow tracking
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="SVR"):
    # Log parameter grid
    param_grid = {
        "kernel": ["linear", "rbf"],
        "C": [1, 10, 77, 100],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1]
    }
    mlflow.log_params({"param_grid": str(param_grid)})
    
    # Grid search with 5-fold CV
    svr = SVR()
    grid = GridSearchCV(svr, param_grid, scoring="neg_root_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_
    
    # Log ALL tested parameter combinations
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(run_name=f"SVR_combo_{i}", nested=True):
            for param, value in params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("cv_mean_score", -grid.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std_score", grid.cv_results_['std_test_score'][i])
    
    # Log best hyperparameters
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # Make predictions
    train_preds_svm = best_svm.predict(X_train)
    val_preds_svm   = best_svm.predict(X_val)
    test_preds_svm  = best_svm.predict(X_test)
    
    # Calculate metrics
    train_rmse_svm = np.sqrt(mean_squared_error(y_train, train_preds_svm))
    val_rmse_svm   = np.sqrt(mean_squared_error(y_val, val_preds_svm))
    test_rmse_svm  = np.sqrt(mean_squared_error(y_test, test_preds_svm))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_svm)
    mlflow.log_metric("val_rmse", val_rmse_svm)
    mlflow.log_metric("test_rmse", test_rmse_svm)
    mlflow.log_metric("cv_best_score", -grid.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(best_svm, "svr_model")
    
    print("Best SVR params:", grid.best_params_)
    print("Train SVM RMSE:", train_rmse_svm)
    print("Validation SVM RMSE:", val_rmse_svm)
    print("Test SVM RMSE:", test_rmse_svm)

# COMMAND ----------

# Save SVR model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
joblib.dump(best_svm, 'svr_best.pkl')
print('SVR model saved as svr_best.pkl')

# Bin predictions and true values for confusion matrix
bins = 5
val_bins_pred = pd.cut(val_preds_svm, bins, labels=False)
val_bins_true = pd.cut(y_val, bins, labels=False)
cm = confusion_matrix(val_bins_true, val_bins_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Bin')
plt.ylabel('Actual Bin')
plt.title('SVR Validation Confusion Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Tree

# COMMAND ----------

# DBTITLE 1,Step 22: Decision Tree Regression (Hyperparameter Tuning + Evaluation)
# Step 22: Fit Decision Tree Regressor with grid search + MLflow tracking
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="Decision_Tree"):
    # Log parameter grid
    param_grid = {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    mlflow.log_params({"param_grid": str(param_grid)})
    
    # Grid search with 5-fold CV
    tree = DecisionTreeRegressor(random_state=42)
    grid = GridSearchCV(tree, param_grid, scoring="neg_root_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)
    tree_model = grid.best_estimator_
    
    # Log ALL tested parameter combinations
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(run_name=f"DecisionTree_combo_{i}", nested=True):
            for param, value in params.items():
                mlflow.log_param(param, str(value))
            mlflow.log_metric("cv_mean_score", -grid.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std_score", grid.cv_results_['std_test_score'][i])
    
    # Log best hyperparameters
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", str(value))
    
    # Make predictions
    train_preds_tree = tree_model.predict(X_train)
    val_preds_tree   = tree_model.predict(X_val)
    test_preds_tree  = tree_model.predict(X_test)
    
    # Calculate metrics
    train_rmse_tree = np.sqrt(mean_squared_error(y_train, train_preds_tree))
    val_rmse_tree   = np.sqrt(mean_squared_error(y_val, val_preds_tree))
    test_rmse_tree  = np.sqrt(mean_squared_error(y_test, test_preds_tree))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_tree)
    mlflow.log_metric("val_rmse", val_rmse_tree)
    mlflow.log_metric("test_rmse", test_rmse_tree)
    mlflow.log_metric("cv_best_score", -grid.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(tree_model, "decision_tree_model")
    
    print("Best Tree params:", grid.best_params_)
    print("Train Tree RMSE:", train_rmse_tree)
    print("Validation Tree RMSE:", val_rmse_tree)
    print("Test Tree RMSE:", test_rmse_tree)

# COMMAND ----------

# Save Decision Tree model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
joblib.dump(tree_model, 'dt_best.pkl')
print('Decision Tree model saved as dt_best.pkl')

# Bin predictions and true values for confusion matrix
bins = 5
val_bins_pred = pd.cut(val_preds_tree, bins, labels=False)
val_bins_true = pd.cut(y_val, bins, labels=False)
cm = confusion_matrix(val_bins_true, val_bins_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted Bin')
plt.ylabel('Actual Bin')
plt.title('Decision Tree Validation Confusion Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forests

# COMMAND ----------

# MAGIC %md
# MAGIC ### Very Slow

# COMMAND ----------

# DBTITLE 1,Step 23: Random Forest Regression (Hyperparameter Tuning + Evaluation)
# Step 23: Fit Random Forest Regressor via grid search + MLflow tracking
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="Random_Forest"):
    # Log parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    mlflow.log_params({"param_grid": str(param_grid)})
    
    # Grid search with 5-fold CV
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, scoring="neg_root_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)
    rf_model = grid.best_estimator_
    
    # Log ALL tested parameter combinations
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(run_name=f"RandomForest_combo_{i}", nested=True):
            for param, value in params.items():
                mlflow.log_param(param, str(value))
            mlflow.log_metric("cv_mean_score", -grid.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std_score", grid.cv_results_['std_test_score'][i])
    
    # Log best hyperparameters
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", str(value))
    
    # Make predictions
    train_preds_rf = rf_model.predict(X_train)
    val_preds_rf   = rf_model.predict(X_val)
    test_preds_rf  = rf_model.predict(X_test)
    
    # Calculate metrics
    train_rmse_rf = np.sqrt(mean_squared_error(y_train, train_preds_rf))
    val_rmse_rf   = np.sqrt(mean_squared_error(y_val, val_preds_rf))
    test_rmse_rf  = np.sqrt(mean_squared_error(y_test, test_preds_rf))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_rf)
    mlflow.log_metric("val_rmse", val_rmse_rf)
    mlflow.log_metric("test_rmse", test_rmse_rf)
    mlflow.log_metric("cv_best_score", -grid.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    print("Best RF params:", grid.best_params_)
    print("Train RF RMSE:", train_rmse_rf)
    print("Validation RF RMSE:", val_rmse_rf)
    print("Test RF RMSE:", test_rmse_rf)

# COMMAND ----------

# Save Random Forest model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
joblib.dump(rf_model, 'rf_best.pkl')
print('Random Forest model saved as rf_best.pkl')

# Bin predictions and true values for confusion matrix
bins = 5
val_bins_pred = pd.cut(val_preds_rf, bins, labels=False)
val_bins_true = pd.cut(y_val, bins, labels=False)
cm = confusion_matrix(val_bins_true, val_bins_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.xlabel('Predicted Bin')
plt.ylabel('Actual Bin')
plt.title('Random Forest Validation Confusion Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Linear Regression

# COMMAND ----------

# DBTITLE 1,Step 24: Linear Regression (Hyperparameter Tuning + Evaluation)
# Step 24: Fit and evaluate Linear Regression with GridSearchCV + MLflow tracking
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="Linear_Regression"):
    # Log parameter grid
    param_grid = {
        "fit_intercept": [True, False]
    }
    mlflow.log_params({"param_grid": str(param_grid)})
    
    # Grid search with 5-fold CV
    lr = LinearRegression()
    grid = GridSearchCV(lr, param_grid, scoring="neg_root_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)
    lr_model = grid.best_estimator_
    
    # Log ALL tested parameter combinations
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(run_name=f"LinearRegression_combo_{i}", nested=True):
            for param, value in params.items():
                mlflow.log_param(param, str(value))
            mlflow.log_metric("cv_mean_score", -grid.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std_score", grid.cv_results_['std_test_score'][i])
    
    # Log best hyperparameters
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", str(value))
    
    # Make predictions
    train_preds_lr = lr_model.predict(X_train)
    val_preds_lr   = lr_model.predict(X_val)
    test_preds_lr  = lr_model.predict(X_test)
    
    # Calculate metrics
    train_rmse_lr = np.sqrt(mean_squared_error(y_train, train_preds_lr))
    val_rmse_lr   = np.sqrt(mean_squared_error(y_val, val_preds_lr))
    test_rmse_lr  = np.sqrt(mean_squared_error(y_test, test_preds_lr))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_lr)
    mlflow.log_metric("val_rmse", val_rmse_lr)
    mlflow.log_metric("test_rmse", test_rmse_lr)
    mlflow.log_metric("cv_best_score", -grid.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "linear_regression_model")
    
    print("Best Linear Regression params:", grid.best_params_)
    print("Train LR RMSE:", train_rmse_lr)
    print("Validation LR RMSE:", val_rmse_lr)
    print("Test LR RMSE:", test_rmse_lr)
    
    # Find coefficients for top 5 features
    feature_names = ["ticker_id"] + features
    coef = lr_model.coef_[:len(feature_names)]
    top_idx = np.argsort(np.abs(coef))[-5:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_coefs = coef[top_idx]
    for f, c in zip(top_features, top_coefs):
        print(f"Feature: {f}, Coefficient: {c}")

# COMMAND ----------

# Save Linear Regression model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
joblib.dump(lr_model, 'lr_best.pkl')
print('Linear Regression model saved as lr_best.pkl')

# Bin predictions and true values for confusion matrix
bins = 5
val_bins_pred = pd.cut(val_preds_lr, bins, labels=False)
val_bins_true = pd.cut(y_val, bins, labels=False)
cm = confusion_matrix(val_bins_true, val_bins_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='gray')
plt.xlabel('Predicted Bin')
plt.ylabel('Actual Bin')
plt.title('Linear Regression Validation Confusion Matrix')
plt.show()

# COMMAND ----------

# DBTITLE 1,Step 25: Install TensorFlow & Keras Tuner
# Step 25: Install TensorFlow and Keras Tuner for deep learning modeling
%pip install tensorflow
%pip install keras-tuner

# COMMAND ----------

# MAGIC %md
# MAGIC # TenserFlow MLP

# COMMAND ----------

# DBTITLE 1,Step 26: TenserFlow MLP (Keras Tuner)
# Step 26: Build, tune, and evaluate deep learning MLP using KerasTuner + MLflow tracking
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras_tuner as kt
import mlflow
import mlflow.keras

with mlflow.start_run(run_name="MLP_Neural_Network"):
    def build_nn_model(hp):
        # KerasTuner hyperparameter search for architecture
        model = Sequential()
        model.add(Dense(
            hp.Int('hidden1', min_value=16, max_value=512, step=32),
            activation='relu',
            input_dim=X_train.shape[1]
        ))
        model.add(Dense(hp.Int('hidden2', min_value=64, max_value=128, step=64), activation='relu'))
        model.add(Dense(hp.Int('hidden3', min_value=32, max_value=64, step=32), activation='relu'))
        model.add(Dense(hp.Int('hidden4', min_value=16, max_value=32, step=16), activation='relu'))
        if hp.Boolean('dropout'):
            model.add(Dropout(hp.Float('dropout_rate', 0.0, 0.5, step=0.1)))
        model.add(Dense(1, activation='linear'))
        model.compile(
            optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
            loss='mse',
            metrics=['mse']
        )
        return model
    
    # Log hyperparameter search space
    mlflow.log_params({
        "hidden1_range": "16-512",
        "hidden2_range": "64-128",
        "hidden3_range": "32-64",
        "hidden4_range": "16-32",
        "optimizer_choices": "adam,rmsprop",
        "max_trials": 20
    })
    
    param_search = kt.RandomSearch(
        build_nn_model,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        directory='keras_tuner_MLP',
        project_name='mlp'
    )
    param_search.search(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=0)
    best_nn = param_search.get_best_models(num_models=1)[0]
    
    # Log ALL tested trials
    all_trials = param_search.oracle.get_best_trials(num_trials=20)
    for i, trial in enumerate(all_trials):
        with mlflow.start_run(run_name=f"MLP_trial_{i}", nested=True):
            for param, value in trial.hyperparameters.values.items():
                mlflow.log_param(param, str(value))
            mlflow.log_metric("val_loss", trial.score if trial.score else 0)
    
    # Log best hyperparameters
    best_hps = param_search.get_best_hyperparameters()[0]
    for param in ['hidden1', 'hidden2', 'hidden3', 'hidden4', 'dropout', 'optimizer']:
        try:
            mlflow.log_param(f"best_{param}", best_hps.get(param))
        except:
            pass
    if best_hps.get('dropout'):
        mlflow.log_param("best_dropout_rate", best_hps.get('dropout_rate'))
    
    # Make predictions
    train_preds_nn = best_nn.predict(X_train)
    val_preds_nn   = best_nn.predict(X_val)
    test_preds_nn  = best_nn.predict(X_test)
    
    # Calculate metrics
    train_rmse_nn = np.sqrt(mean_squared_error(y_train, train_preds_nn))
    val_rmse_nn   = np.sqrt(mean_squared_error(y_val, val_preds_nn))
    test_rmse_nn  = np.sqrt(mean_squared_error(y_test, test_preds_nn))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_nn)
    mlflow.log_metric("val_rmse", val_rmse_nn)
    mlflow.log_metric("test_rmse", test_rmse_nn)
    
    # Log model
    mlflow.keras.log_model(best_nn, "mlp_model")
    
    print("Best NN params:", best_hps.values)
    print("Train NN RMSE:", train_rmse_nn)
    print("Validation NN RMSE:", val_rmse_nn)
    print("Test NN RMSE:", test_rmse_nn)

# COMMAND ----------

# DBTITLE 1,MLP Model Save & Confusion Matrix
# Save MLP model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
try:
    joblib.dump(best_nn, 'mlp_best.pkl')
    print('MLP model saved as mlp_best.pkl')
except Exception as e:
    print(f'Could not save MLP model: {e}')

# Bin predictions and true values for confusion matrix
bins = 5
try:
    val_bins_pred = pd.cut(val_preds_nn.flatten(), bins, labels=False)
    val_bins_true = pd.cut(y_val, bins, labels=False)
    cm = confusion_matrix(val_bins_true, val_bins_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted Bin')
    plt.ylabel('Actual Bin')
    plt.title('MLP Validation Confusion Matrix')
    plt.show()
except Exception as e:
    print(f'Could not plot MLP confusion matrix: {e}')


# COMMAND ----------

# DBTITLE 1,Step 27: (Optionally) Summarize MLP Model Architecture
# Step 27: Print summary of MLP architecture found by KerasTuner
best_nn.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGboost

# COMMAND ----------

# DBTITLE 1,Step 28: Install XGBoost
# Step 28: Install XGBoost for model comparison
%pip install xgboost

# COMMAND ----------

# DBTITLE 1,Step 29: XGBoost Regression (Grid Search + Evaluation)
# Step 29: Fit and tune XGBoost model + MLflow tracking
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.xgboost

with mlflow.start_run(run_name="XGBoost"):
    # Log parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    mlflow.log_params({"param_grid": str(param_grid)})
    
    # Grid search with 5-fold CV
    xgb = XGBRegressor(random_state=42)
    grid = GridSearchCV(xgb, param_grid, scoring="neg_root_mean_squared_error", cv=5)
    grid.fit(X_train, y_train)
    xgb_model = grid.best_estimator_
    
    # Log ALL tested parameter combinations
    for i, params in enumerate(grid.cv_results_['params']):
        with mlflow.start_run(run_name=f"XGBoost_combo_{i}", nested=True):
            for param, value in params.items():
                mlflow.log_param(param, str(value))
            mlflow.log_metric("cv_mean_score", -grid.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std_score", grid.cv_results_['std_test_score'][i])
    
    # Log best hyperparameters
    for param, value in grid.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # Make predictions
    train_preds_xgb = xgb_model.predict(X_train)
    val_preds_xgb   = xgb_model.predict(X_val)
    test_preds_xgb  = xgb_model.predict(X_test)
    
    # Calculate metrics
    train_rmse_xgb = np.sqrt(mean_squared_error(y_train, train_preds_xgb))
    val_rmse_xgb   = np.sqrt(mean_squared_error(y_val, val_preds_xgb))
    test_rmse_xgb  = np.sqrt(mean_squared_error(y_test, test_preds_xgb))
    
    # Log metrics
    mlflow.log_metric("train_rmse", train_rmse_xgb)
    mlflow.log_metric("val_rmse", val_rmse_xgb)
    mlflow.log_metric("test_rmse", test_rmse_xgb)
    mlflow.log_metric("cv_best_score", -grid.best_score_)
    
    # Log model
    mlflow.xgboost.log_model(xgb_model, "xgboost_model")
    
    print("Best XGB params:", grid.best_params_)
    print("Train XGBoost RMSE:", train_rmse_xgb)
    print("Validation XGBoost RMSE:", val_rmse_xgb)
    print("Test XGBoost RMSE:", test_rmse_xgb)

# COMMAND ----------

# Save XGBoost model and plot confusion matrix
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save model
try:
    joblib.dump(xgb_model, 'xgb_best.pkl')
    print('XGBoost model saved as xgb_best.pkl')
except Exception as e:
    print(f'Could not save XGBoost model: {e}')

# Bin predictions and true values for confusion matrix
bins = 5
try:
    val_bins_pred = pd.cut(val_preds_xgb, bins, labels=False)
    val_bins_true = pd.cut(y_val, bins, labels=False)
    cm = confusion_matrix(val_bins_true, val_bins_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis')
    plt.xlabel('Predicted Bin')
    plt.ylabel('Actual Bin')
    plt.title('XGBoost Validation Confusion Matrix')
    plt.show()
except Exception as e:
    print(f'Could not plot XGBoost confusion matrix: {e}')


# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation

# COMMAND ----------

# DBTITLE 1,Step 30: Evaluate Model Results: Actual vs Predicted Scatter
# Step 30: Visualize Actual vs Predicted for all trained models
import matplotlib.pyplot as plt

# Collect predictions for each model
model_preds = {}
# XGBoost
try:
    model_preds['XGBoost'] = val_preds_xgb
except:
    pass
# Neural Net
try:
    model_preds['Neural Net'] = val_preds_nn.flatten()
except Exception:
    pass
# Ridge Regression
try:
    model_preds['Ridge'] = val_preds_ridge
except Exception:
    pass
# SVM
try:
    model_preds['SVM'] = val_preds_svm
except Exception:
    pass
# Decision Tree
try:
    model_preds['Decision Tree'] = val_preds_tree
except Exception:
    pass
# Random Forest
try:
    model_preds['Random Forest'] = val_preds_rf
except Exception:
    pass
# Linear Regression
try:
    model_preds['Linear Regression'] = val_preds_lr
except Exception:
    pass

# Plot actual vs predicted for each model
fig, axes = plt.subplots(len(model_preds), 1, figsize=(8, 4 * len(model_preds)), sharex=True)
if len(model_preds) == 1:
    axes = [axes]
for ax, (model_name, preds) in zip(axes, model_preds.items()):
    ax.scatter(y_val, preds, alpha=0.5)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name}: Predictions vs Actuals')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Importance

# COMMAND ----------


import numpy as np

# Combine numeric and text feature names
feature_names = num_features

# Use a small batch for speed (e.g., first 100 samples)
n_samples = min(500, X_test.shape[0])
X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
X_batch = tf.convert_to_tensor(X_test_dense[:n_samples], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(X_batch)
    preds = best_nn(X_batch)

grads = tape.gradient(preds, X_batch)  # shape: (n_samples, n_features)
feature_importance = np.mean(np.abs(grads.numpy()), axis=0)  # average over samples

# Top 20 features
top_idx = np.argsort(feature_importance)[-20:][::-1]
top_features = [feature_names[i] for i in top_idx]
top_importances = feature_importance[top_idx]

importance_df = pd.DataFrame({'Feature': top_features, 'Importance': top_importances})
display(importance_df)

plt.figure(figsize=(10,6))
plt.bar(top_features, top_importances)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Gradient Importance')
plt.title('MLP (Neural Net) Top 20 Feature Importance (Gradient-based)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Importance for Each Model
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse

# Feature names (numeric + text features)
feature_names = ["ticker_id"] + features

model_importances = {}

# Ridge Regression
try:
    model_importances['Ridge'] = np.abs(ridge_model.coef_[:len(feature_names)])
except Exception:
    pass

# SVM (linear kernel only)
try:
    if hasattr(best_svm, 'coef_'):  # Fixed: was checking svm_model instead of best_svm
        importances = best_svm.coef_
        if issparse(importances):
            importances = importances.toarray().flatten()
        else:
            importances = importances.flatten()
        model_importances['SVM'] = np.abs(importances[:len(feature_names)])
except Exception:
    pass

# XGBoost
try:
    model_importances['XGBoost'] = xgb_model.feature_importances_[:len(feature_names)]
except Exception:
    pass

# Decision Tree
try:
    model_importances['Decision Tree'] = tree_model.feature_importances_[:len(feature_names)]
except Exception:
    pass

# Random Forest
try:
    model_importances['Random Forest'] = rf_model.feature_importances_[:len(feature_names)]
except Exception:
    pass

# Linear Regression
try:
    model_importances['Linear Regression'] = np.abs(lr_model.coef_[:len(feature_names)])
except Exception:
    pass


# Plot top 20 feature importance for each model
fig, axes = plt.subplots(len(model_importances), 1, figsize=(12, 5 * len(model_importances)), sharex=False)
if len(model_importances) == 1:
    axes = [axes]

for ax, (model_name, importances) in zip(axes, model_importances.items()):
    # Only plot if importances is 1D
    if importances is not None and importances.ndim == 1:
        # Convert sparse matrix to dense if needed
        if issparse(importances):
            importances = importances.toarray().flatten()
        # Get top 20 features
        top_idx = np.argsort(importances)[-20:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]
        ax.bar(top_features, top_importances)
        ax.set_ylabel('Importance')
        ax.set_xticklabels(top_features, rotation=45, ha='right')
    else:
        pass
    ax.set_title(f'{model_name} Top 20 Feature Importance')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Compare to non text models (numeric-only)
# Remove sentiment-related features for numeric-only modeling
non_sentiment_features = [f for f in num_features if not ('sentiment' in f.lower())]

scaler_num = StandardScaler()
X_num_train = scaler_num.fit_transform(train_df[non_sentiment_features])
X_num_test  = scaler_num.transform(test_df[non_sentiment_features])
y_num_train = y_train
y_num_test  = y_test

print("Numeric-only shape:", X_num_train.shape, X_num_test.shape)

# COMMAND ----------

# DBTITLE 1,Numeric Ridge Regression
import numpy as np
ridge_num_model = Ridge(alpha=1.0)
ridge_num_model.fit(X_num_train, y_num_train)

train_preds_ridge = ridge_num_model.predict(X_num_train)
test_preds_ridge  = ridge_num_model.predict(X_num_test)

train_rmse_ridge_n = np.sqrt(mean_squared_error(y_num_train, train_preds_ridge))
test_rmse_ridge_n  = np.sqrt(mean_squared_error(y_num_test, test_preds_ridge))

print("Train Ridge RMSE:", train_rmse_ridge)
print("Test Ridge RMSE:", test_rmse_ridge)

# COMMAND ----------

# DBTITLE 1,Numeric SVM Regression
from sklearn import svm
svm_num_model = svm.SVR(kernel="linear")
svm_num_model.fit(X_num_train, y_num_train)

train_preds_svm = svm_num_model.predict(X_num_train)
test_preds_svm  = svm_num_model.predict(X_num_test)

train_rmse_svm_n = np.sqrt(mean_squared_error(y_num_train, train_preds_svm))
test_rmse_svm_n  = np.sqrt(mean_squared_error(y_num_test, test_preds_svm))

print("Train SVM RMSE:", train_rmse_svm)
print("Test SVM RMSE:", test_rmse_svm)

# COMMAND ----------

# DBTITLE 1,Numeric Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_num_model = DecisionTreeRegressor()
tree_num_model.fit(X_num_train, y_num_train)

train_preds_tree = tree_num_model.predict(X_num_train)
test_preds_tree  = tree_num_model.predict(X_num_test)

train_rmse_tree_n = np.sqrt(mean_squared_error(y_num_train, train_preds_tree))
test_rmse_tree_n  = np.sqrt(mean_squared_error(y_num_test, test_preds_tree))

print("Train Tree RMSE:", train_rmse_tree)
print("Test Tree RMSE:", test_rmse_tree)

# COMMAND ----------

# DBTITLE 1,Numeric Random Forest Regression
'''from sklearn.ensemble import RandomForestRegressor
rf_num_model = RandomForestRegressor(n_estimators=100)
rf_num_model.fit(X_num_train, y_num_train)

train_preds_rf = rf_num_model.predict(X_num_train)
test_preds_rf  = rf_num_model.predict(X_num_test)

train_rmse_rf_n = np.sqrt(mean_squared_error(y_num_train, train_preds_rf))
test_rmse_rf_n  = np.sqrt(mean_squared_error(y_num_test, test_preds_rf))

print("Train RF RMSE:", train_rmse_rf)
print("Test RF RMSE:", test_rmse_rf)'''

# COMMAND ----------

# DBTITLE 1,Numeric Linear Regression
from sklearn.linear_model import LinearRegression
lr_num_model = LinearRegression()
lr_num_model.fit(X_num_train, y_num_train)
train_preds_lr = lr_num_model.predict(X_num_train)
test_preds_lr  = lr_num_model.predict(X_num_test)

train_rmse_lr_n = np.sqrt(mean_squared_error(y_num_train, train_preds_lr))
test_rmse_lr_n  = np.sqrt(mean_squared_error(y_num_test, test_preds_lr))

print("Train LR RMSE:", train_rmse_lr)
print("Test LR RMSE:", test_rmse_lr)

# COMMAND ----------

# DBTITLE 1,Numeric Neural Net Regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

nn_num_model = Sequential()
nn_num_model.add(Dense(256, input_dim=X_num_train.shape[1], activation="relu"))
nn_num_model.add(Dense(128, activation="relu"))
nn_num_model.add(Dense(64, activation="relu"))
nn_num_model.add(Dense(32, activation="relu"))
nn_num_model.add(Dense(1, activation="linear"))

nn_num_model.compile(loss="mse", optimizer="adam", metrics=["mse"])

history_num = nn_num_model.fit(X_num_train, y_num_train, epochs=100, batch_size=32, validation_split=0.2)
train_preds_nn = nn_num_model.predict(X_num_train)
test_preds_nn  = nn_num_model.predict(X_num_test)

train_rmse_nn_n = np.sqrt(mean_squared_error(y_num_train, train_preds_nn))
test_rmse_nn_n  = np.sqrt(mean_squared_error(y_num_test, test_preds_nn))

print("Train NN RMSE:", train_rmse_nn)
print("Test NN RMSE:", test_rmse_nn)

# COMMAND ----------

# MAGIC %md
# MAGIC # Comparison of models with and without text features

# COMMAND ----------

# DBTITLE 1,Model RMSE Summary
# Collect RMSE values for text+numeric and numeric-only models
summary = []


summary.append(["Ridge (text+num)", train_rmse_ridge, test_rmse_ridge])
summary.append(["Ridge (num only)", train_rmse_ridge_n, test_rmse_ridge_n])
summary.append(["SVM (text+num)", train_rmse_svm, test_rmse_svm])
summary.append(["SVM (num only)", train_rmse_svm_n, test_rmse_svm_n])
summary.append(["Decision Tree (text+num)", train_rmse_tree, test_rmse_tree])
summary.append(["Decision Tree (num only)", train_rmse_tree_n, test_rmse_tree_n])
try:
    summary.append(["Random Forest (text+num)", train_rmse_rf, test_rmse_rf])
except:
    pass
try:
    summary.append(["Random Forest (num only)", train_rmse_rf_n, test_rmse_rf_n])
except:
    pass
summary.append(["Linear Regression (text+num)", train_rmse_lr, test_rmse_lr])
summary.append(["Linear Regression (num only)", train_rmse_lr_n, test_rmse_lr_n])
summary.append(["Neural Net (text+num)", train_rmse_nn, test_rmse_nn])
summary.append(["Neural Net (num only)", train_rmse_nn_n, test_rmse_nn_n])


import pandas as pd
summary_df = pd.DataFrame(summary, columns=["Model", "Train RMSE", "Test RMSE"])
display(summary_df)


# COMMAND ----------

summary_df.to_csv("results.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC - Moving Forward will likely use SVM as it is showing the greatest improvement with text data
# MAGIC - Make sure to update GitHub after each change
# MAGIC