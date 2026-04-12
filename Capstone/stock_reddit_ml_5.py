# Databricks notebook source
# MAGIC %md
# MAGIC # Stock + Reddit Multi-Stock ML Pipeline (Updated)
# MAGIC
# MAGIC This version:
# MAGIC - Supports **multiple stocks**
# MAGIC - Reuses the same Reddit text per day across tickers
# MAGIC - Creates **multiple prediction horizons**
# MAGIC - Avoids data leakage
# MAGIC

# COMMAND ----------

# Imports
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Stock Data (Multi-Ticker)

# COMMAND ----------

# numerical_data.csv must include: date, ticker, Close
df = pd.read_csv("numerical_data.csv")

df["date"] = pd.to_datetime(df["date"]).dt.date
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stock Feature Engineering

# COMMAND ----------

# Per-ticker returns
df["ret_1"] = df.groupby("ticker")["Close"].pct_change(1)
df["ret_3"] = df.groupby("ticker")["Close"].pct_change(3)
df["ret_5"] = df.groupby("ticker")["Close"].pct_change(5)

# Volatility
df["vol_10"] = df.groupby("ticker")["ret_1"].rolling(10).std().reset_index(0,drop=True)

# SMA ratio
sma_10 = df.groupby("ticker")["Close"].rolling(10).mean().reset_index(0,drop=True)
df["sma_ratio"] = df["Close"] / sma_10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Targets (Multiple Horizons)

# COMMAND ----------

# Predict forward returns
df["target_1d"] = df.groupby("ticker")["Close"].pct_change(-1)
df["target_3d"] = df.groupby("ticker")["Close"].pct_change(-3)
df["target_5d"] = df.groupby("ticker")["Close"].pct_change(-5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Aggregate Reddit Text

# COMMAND ----------

df_reddit = pd.read_excel("combined_text_by_date.xlsx")
df_reddit["date"] = pd.to_datetime(df_reddit["date"]).dt.date

# One text blob per day
df_reddit_daily = (
    df_reddit.groupby("date")["text"]
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Stock + Reddit

# COMMAND ----------

df = df.merge(df_reddit_daily, on="date", how="left")
df["text"] = df["text"].fillna("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Cleaning

# COMMAND ----------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Dataset

# COMMAND ----------

features = ["ret_1", "ret_3", "ret_5", "vol_10", "sma_ratio"]
target = "target_1d"

df_model = df.dropna(subset=features + [target]).reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vectorize Text + Combine Features

# COMMAND ----------

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    min_df=3
)

X_text = vectorizer.fit_transform(df_model["clean_text"])

scaler = StandardScaler()
X_num = scaler.fit_transform(df_model[features])

X = hstack([X_num, X_text])
y = df_model[target].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

model = Ridge(alpha=1.0)
model.fit(X, y)

preds = model.predict(X)
rmse = mean_squared_error(y, preds, squared=False)

print("Train RMSE:", rmse)