# Databricks notebook source
# MAGIC %md
# MAGIC # Preprocessing Steps
# MAGIC
# MAGIC This notebook:
# MAGIC - Downloads **multi-ticker stock data** automatically
# MAGIC - Downloads Text Data
# MAGIC - Adds **ticker awareness** so all stocks contribute
# MAGIC - Uses **time-based train/test split** (no leakage)
# MAGIC - Includes diagnostics to verify ticker balance
# MAGIC

# COMMAND ----------

!pip install yfinance openpyxl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download Stock Data

# COMMAND ----------

import yfinance as yf
import pandas as pd

START = "2025-10-01"
END = "2026-02-08"

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

# DBTITLE 1,Stock Feature Engineering (Per Ticker)
df["ret_1"] = df.groupby("ticker")["Close"].pct_change(1)
df["ret_3"] = df.groupby("ticker")["Close"].pct_change(3)
df["ret_5"] = df.groupby("ticker")["Close"].pct_change(5)

df["vol_10"] = (
    df.groupby("ticker")["ret_1"]
    .rolling(10)
    .std()
    .reset_index(0, drop=True)
)

sma_10 = (
    df.groupby("ticker")["Close"]
    .rolling(10)
    .mean()
    .reset_index(0, drop=True)
)

df["sma_ratio"] = df["Close"] / sma_10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Prediction Targets

# COMMAND ----------

df["target_1d"] = df.groupby("ticker")["Close"].pct_change(-1)
df["target_3d"] = df.groupby("ticker")["Close"].pct_change(-3)
df["target_5d"] = df.groupby("ticker")["Close"].pct_change(-5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load & Aggregate Reddit Text

# COMMAND ----------

import praw
SUBREDDIT = "stocks"

try:
    reddit = praw.Reddit(
        client_id="Your-ID",
        client_secret="Your-Secret",
        user_agent="Your-Agent"
    )
    posts = []
    for post in reddit.subreddit(SUBREDDIT).new(limit=None):
        post_date = datetime.utcfromtimestamp(post.created_utc).date()
        if START <= str(post_date) < END:
            post.comments.replace_more(limit=0)
            comments_text = " ".join([c.body for c in post.comments.list()])
            posts.append({
                "date": post_date,
                "text": post.title + " " + post.selftext + " " + comments_text
        })
    reddit_daily = pd.DataFrame(posts)
except:
    print("Error With Reddit API")

reddit_daily.head()

# COMMAND ----------

reddit_daily['date'] = pd.to_datetime(reddit_daily['date']).dt.normalize()

# Combine text by date
df_combined_text = (
    reddit_daily.groupby('date')['text']
    .apply(lambda x: ' '.join(x.dropna().astype(str)))
    .reset_index()
)

# Clean text (still good practice)
df_combined_text['text'] = (
    df_combined_text['text']
    .apply(lambda x: re.sub(r'[\x00-\x1F\x7F\u2028\u2029]', ' ', x))
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

# Excel-friendly date format
df_combined_text['date'] = df_combined_text['date'].dt.strftime('%Y-%m-%d')
reddit_daily = df_combined_text[['date', 'text']]
reddit_daily.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Merge Stock + Reddit Data

# COMMAND ----------

df = df.merge(reddit_daily, on="date", how="left")
df["text"] = df["text"].fillna("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Clean Text

# COMMAND ----------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Final Modeling Dataset

# COMMAND ----------

features = ["ret_1", "ret_3", "ret_5", "vol_10", "sma_ratio"]
target = "target_1d"

df_model = df.dropna(subset=features + [target]).reset_index(drop=True)

# Drop rows without text
df_model = df_model[df_model["text"].str.strip() != ""]

# Add ticker awareness
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

print("Rows per ticker:")
print(df_model["ticker"].value_counts())

print("\nDate range:")
print(df_model["date"].min(), "->", df_model["date"].max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Time-Based Train/Test Split

# COMMAND ----------

split_date = df_model["date"].quantile(0.8)

train_df = df_model[df_model["date"] <= split_date]
test_df  = df_model[df_model["date"] > split_date]

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Vectorize Text (Train Only)

# COMMAND ----------

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    min_df=3
)

X_text_train = vectorizer.fit_transform(train_df["clean_text"])
X_text_test  = vectorizer.transform(test_df["clean_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Scale Numeric Features (Train Only)

# COMMAND ----------

num_features = ["ticker_id"] + features

scaler = StandardScaler()
X_num_train = scaler.fit_transform(train_df[num_features])
X_num_test  = scaler.transform(test_df[num_features])

X_train = hstack([X_num_train, X_text_train])
X_test  = hstack([X_num_test, X_text_test])

y_train = train_df[target].values
y_test  = test_df[target].values
