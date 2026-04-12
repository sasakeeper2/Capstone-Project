# Databricks notebook source
# DBTITLE 1,Cell 1
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

df = pd.read_csv("df_model.csv")
df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

# DBTITLE 1,Cell 3
import matplotlib.dates as mdates

# Convert date column to datetime
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# Aggregate by day (if needed)
daily_df = df.groupby(df['date'].dt.date)['Close'].mean().reset_index()
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Plot only the last 100 days
plot_df = daily_df.tail(100)

plt.figure(figsize=(10, 5))
plt.plot(plot_df['date'], plot_df['Close'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Over Time (Last 100 Days)')
plt.xticks(rotation=45)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC   - This shows that the price is not linear
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - This is important because it shows that linear models likely will not work
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This confirms expectations as stock prices are notorious for rising and falling
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - The next question is "If linear models won't work, what will."

# COMMAND ----------

import matplotlib.dates as mdates

# Convert date column to datetime
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# Aggregate by day (if needed)
daily_df = df.groupby(df['date'].dt.date)['Volume'].mean().reset_index()
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Plot only the last 100 days
plot_df = daily_df.tail(100)

plt.figure(figsize=(10, 5))
plt.plot(plot_df['date'], plot_df['Volume'])
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume Over Time (Last 100 Days)')
plt.xticks(rotation=45)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.dates as mdates

# Convert date column to datetime
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# Aggregate by day (if needed)
daily_df = df.groupby(df['date'].dt.date)['vol_10'].mean().reset_index()
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Plot only the last 100 days
plot_df = daily_df.tail(100)

plt.figure(figsize=(10, 5))
plt.plot(plot_df['date'], plot_df['vol_10'])
plt.xlabel('Date')
plt.ylabel('Vol_10')
plt.title('Vol_10 Over Time (Last 100 Days)')
plt.xticks(rotation=45)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.show()

# COMMAND ----------

from pandas.plotting import autocorrelation_plot

plt.figure()
autocorrelation_plot(df['target_1d'].dropna())
plt.title("Autocorrelation of target_1d")
plt.show()

# COMMAND ----------

from pandas.plotting import autocorrelation_plot

plt.figure()
autocorrelation_plot(df['target_3d'].dropna())
plt.title("Autocorrelation of target_3d")
plt.show()

# COMMAND ----------

from pandas.plotting import autocorrelation_plot

plt.figure()
autocorrelation_plot(df['target_5d'].dropna())
plt.title("Autocorrelation of target_5d")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC   - All three of these show that the lagged data is correlated to the price, however it is not as correlated as one would imagine
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - This confirms the idea that the price is not linear as correlation only measures linearity
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This confirms expectations set in the last visualizations
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - The major question that arises is "Is correlation even important?"

# COMMAND ----------

import seaborn as sns

# Find most correlated columns to each target
targets = ['target_1d', 'target_3d', 'target_5d']
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

most_corr_cols = {}
for target in targets:
    if target in corr.columns:
        # Exclude self-correlation and columns with 'target' or 'Close' in the name
        exclude_cols = [col for col in corr.columns if 'target' in col or 'Close' in col]
        corr_to_target = corr[target].drop(exclude_cols).abs().sort_values(ascending=False)
        # Get top 2 most correlated columns for each target
        most_corr_cols[target] = corr_to_target.head(2).index.tolist()

# Flatten and deduplicate columns
plot_columns = set()
for cols in most_corr_cols.values():
    plot_columns.update(cols)
# Exclude targets and close variables from plot_columns
plot_columns = {col for col in plot_columns if 'target' not in col and 'Close' not in col}

# Plot pairplot for most correlated columns (excluding targets and close variables)
sns.pairplot(df[list(plot_columns)].dropna())
plt.suptitle("Pairplot of Most Correlated Features (Excluding Targets and Close)", y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

import numpy as np

numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

# Find top 10 columns with highest absolute correlation (excluding self-correlation)
corr_matrix = corr.abs()
np.fill_diagonal(corr_matrix.values, 0)
top_corr_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
top_cols = set()
for idx, (col1, col2) in enumerate(top_corr_pairs.index):
    top_cols.add(col1)
    top_cols.add(col2)
    if len(top_cols) >= 10:
        break
top_cols = list(top_cols)[:10]
corr_top = corr.loc[top_cols, top_cols]

plt.figure()
plt.imshow(corr_top)
plt.colorbar()
plt.title("Correlation Heatmap (Top 10 Highest Correlated Columns)")
plt.xticks(range(len(corr_top.columns)), corr_top.columns, rotation=90)
plt.yticks(range(len(corr_top.columns)), corr_top.columns)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC   - This again confirms the non linearity due to the lack of meaningful correlation even in the most correlated variables
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - It backs previous ideas
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This once again confirms expectations
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - This actually anwsers questions about if correlation is important. It means that if future models can model the data, then correlation is not particularly important

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

# DBTITLE 1,Cell 11
all_text = ' '.join(df['text'].astype(str).values)

import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords

# Download stopwords if not already present
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Basic cleaning
words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

# Remove short words and stop words
words = [w for w in words if len(w) > 3 and w not in stop_words]

word_counts = Counter(words)
top_words = word_counts.most_common(20)

words_plot = [w[0] for w in top_words]
counts_plot = [w[1] for w in top_words]

plt.figure()
plt.bar(words_plot, counts_plot)
plt.title("Top 20 Most Common Words")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC   - This shows the most used words in our text data
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - This is important because it shows what people are really talking about in our data
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This confirms expectations as it appears people are talking about stocks and things that notoriously effect stocks
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - How does this effect stock data

# COMMAND ----------

import seaborn as sns

# Assume 'sentiment' column exists in df, with values in float
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# Aggregate average sentiment by day
sentiment_daily = df.groupby(df['date'].dt.date)['sentiment'].mean().reset_index()
sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])

plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='sentiment', data=sentiment_daily)
plt.title("Average Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC   - Sentiment is fairly stable
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - It shows a possible lack of meaning in sentiment
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This contradicts expectations as it was expected to follow a similar pattern to the price if it was meaningful
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - Is sentiment meaningful at all?

# COMMAND ----------

import seaborn as sns

# Compute average, max, and min close price across the entire dataset
avg_close = df['Close'].mean()
max_close = df['Close'].max()
min_close = df['Close'].min()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Close'])
plt.axvline(avg_close, color='red', linestyle='--', label=f'Average Close: {avg_close:.2f}')
plt.axvline(max_close, color='green', linestyle=':', label=f'Max Close: {max_close:.2f}')
plt.axvline(min_close, color='blue', linestyle=':', label=f'Min Close: {min_close:.2f}')
plt.title("Boxplot of Close Prices\n(Average, Max, Min Close Highlighted)", fontsize=16)
plt.xlabel("Close Price", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC  - This shows average, max, and min price
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - This is important as it shows outliers and where the outliers are
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This confirms that although there are outliers, most the data is within reasonable bounds
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - Should outliers be removed?

# COMMAND ----------

from mpl_toolkits.mplot3d import Axes3D

# Find top 2 variables most correlated with a target (e.g., 'target_1d')
target = 'target_1d'
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

# Exclude target and 'Close' columns
exclude_cols = [col for col in corr.columns if 'target' in col or 'Close' in col]
corr_to_target = corr[target].drop(exclude_cols).abs().sort_values(ascending=False)
top_vars = corr_to_target.head(2).index.tolist()

# Prepare data for 3D plot
plot_df = df[[top_vars[0], top_vars[1], target]].dropna()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plot_df[top_vars[0]], plot_df[top_vars[1]], plot_df[target], c=plot_df[target], cmap='viridis', alpha=0.7)
ax.set_xlabel(top_vars[0])
ax.set_ylabel(top_vars[1])
ax.set_zlabel(target)
ax.set_title(f"3D Scatter: {top_vars[0]}, {top_vars[1]} vs {target}")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - What does this visualization reveal?
# MAGIC   -The distinct lack of correlation even with extra supporting variables
# MAGIC
# MAGIC - Why is this important?
# MAGIC   - It confirms the lack of correlation
# MAGIC
# MAGIC - Does this confirm or contradict expectations?
# MAGIC   - This confirms everything that has been seen so far
# MAGIC
# MAGIC - What new questions arise?
# MAGIC   - Can a non-linear dataset be modeled?