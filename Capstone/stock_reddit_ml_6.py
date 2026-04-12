# Databricks notebook source
# MAGIC %md
# MAGIC ## Download Stock Data

# COMMAND ----------

!pip install yfinance openpyxl

# COMMAND ----------

# Stock data download cell
# This pulls OHLCV data for multiple tickers over the same time window

import yfinance as yf
import pandas as pd

tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "BAC", "SPY"
]

START_DATE = "2024-01-01"
END_DATE = "2026-02-08"

all_data = []

for ticker in tickers:
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False
    ).reset_index()

    df["ticker"] = ticker
    df = df[["Date", "ticker", "Close", "Volume"]]
    df.rename(columns={"Date": "date"}, inplace=True)

    all_data.append(df)

numerical_data = pd.concat(all_data, ignore_index=True)
numerical_data["date"] = pd.to_datetime(numerical_data["date"]).dt.date

numerical_data.to_csv("numerical_data.csv", index=False)

print("Saved numerical_data.csv with shape:", numerical_data.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Stock Data (Multi-Ticker)

# COMMAND ----------

import pandas as pd
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
# MAGIC ## Stock Feature Engineering

# COMMAND ----------

# DBTITLE 1,Stock Feature Engineering (fix dtype robust)
# Ensure numeric columns are float, coercing errors to NaN
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

df["ret_1"] = df.groupby("ticker")["Close"].pct_change(1)
df["ret_3"] = df.groupby("ticker")["Close"].pct_change(3)
df["ret_5"] = df.groupby("ticker")["Close"].pct_change(5)

df["vol_10"] = df.groupby("ticker")["ret_1"].rolling(10).std().reset_index(0, drop=True)

sma_10 = df.groupby("ticker")["Close"].rolling(10).mean().reset_index(0, drop=True)
df["sma_ratio"] = df["Close"] / sma_10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Targets (Multiple Horizons)

# COMMAND ----------

df["target_1d"] = df.groupby("ticker")["Close"].pct_change(-1)
df["target_3d"] = df.groupby("ticker")["Close"].pct_change(-3)
df["target_5d"] = df.groupby("ticker")["Close"].pct_change(-5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Aggregate Reddit Text

# COMMAND ----------

df_reddit = pd.read_excel("combined_text_by_date.xlsx")
df_reddit["date"] = pd.to_datetime(df_reddit["date"]).dt.date

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
df_model.to_csv("df_model.csv", index=False)

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