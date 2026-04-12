# Databricks notebook source
!pip install openpyxl

# COMMAND ----------


# Imports
import pandas as pd
import numpy as np
import re
import openpyxl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Stock Data

# COMMAND ----------


# Load stock data
df = pd.read_csv("numerical_data.csv")

df["date"] = pd.to_datetime(df["date"]).dt.date
df = df.sort_values("date").reset_index(drop=True)

df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------


# Returns
df["ret_1"] = df["Close"].pct_change(1)
df["ret_5"] = df["Close"].pct_change(5)

# Rolling features
df["vol_10"] = df["ret_1"].rolling(10).std()
df["sma_10"] = df["Close"].rolling(10).mean()
df["sma_30"] = df["Close"].rolling(30).mean()
df["sma_ratio"] = df["sma_10"] / df["sma_30"]

# Target: next-day return
df["target"] = df["ret_1"].shift(-1)

df.head(15)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Aggregate Reddit Data

# COMMAND ----------


# Load Reddit data
df_reddit = pd.read_excel("combined_text_by_date.xlsx")

df_reddit["date"] = pd.to_datetime(df_reddit["date"]).dt.date

# Aggregate all posts per day into one string
df_reddit_daily = (
    df_reddit.groupby("date")["text"]
    .apply(lambda x: " ".join(x.dropna().astype(str)))
    .reset_index()
)

df_reddit_daily.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Stock + Reddit

# COMMAND ----------


df = df.merge(df_reddit_daily, on="date", how="left")

# IMPORTANT: Do NOT drop rows because of missing text
df["text"] = df["text"].fillna("")

df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Cleaning

# COMMAND ----------


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].astype(str).apply(clean_text)

df[["date", "clean_text"]].head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Dataset Cleanup

# COMMAND ----------


# Drop only rows missing NUMERIC features, target, or text
df_model = df.dropna(subset=[
    "ret_1", "ret_5", "vol_10", "sma_ratio", "target", "text"
]).reset_index(drop=True)

# Additionally, drop rows where text is empty
df_model = df_model[df_model["text"].str.strip() != ""]

print("Final rows:", df_model.shape[0])
df_model.head()

df_model.to_csv("model_data.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Features

# COMMAND ----------


numeric_features = ["ret_1", "ret_5", "vol_10", "sma_ratio"]
X_num = df_model[numeric_features].values
y = df_model["target"].values

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)

X_text = vectorizer.fit_transform(df_model["clean_text"])

X = hstack([X_num, X_text])

X.shape


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------


model = Ridge(alpha=1.0)
model.fit(X, y)

preds = model.predict(X)
rmse = mean_squared_error(y, preds, squared=False)

print("Train RMSE:", rmse)
