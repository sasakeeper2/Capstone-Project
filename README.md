# Capstone Project

A stock price prediction capstone using stock market data and Reddit text sentiment.

## Overview

This repository contains a Python-based workflow for:
- downloading multi-ticker stock data,
- engineering time-series features,
- aggregating Reddit text data,
- merging stock and sentiment information,
- training prediction models, and
- visualizing the results.

The main work is located in the `Capstone/` folder.

## Project structure

- `Capstone/Preprocessing.py` - download stock data, build time-series features, collect Reddit text, merge datasets, and prepare model input.
- `Capstone/stock_reddit_7.py` - build a regression pipeline for next-day return prediction.
- `Capstone/EDA.py` - exploratory analysis and time-series visualizations of stock price, volume, and volatility.
- `Capstone/keras_tuner_MLP/` - Keras Tuner output and trial artifacts for neural network experiments.
- `Capstone/combined_text_by_date.xlsx` - aggregated Reddit text by date.
- `Capstone/numerical_data.csv` - cleaned multi-ticker stock data.
- `Capstone/Capstone/` - nested project files and model assets.

## Setup

Recommended Python version: 3.10 or later.

Install the key dependencies:

```powershell
cd Capstone
pip install pandas numpy scikit-learn scipy matplotlib yfinance openpyxl praw
```

If you are running in a notebook environment, the notebooks may install any additional packages automatically.

## Usage

1. Open the `Capstone/` folder in your editor.
2. Run `Preprocessing.py` to build the dataset and merge stock + Reddit text.
3. Run `stock_reddit_7.py` to train a baseline regression model.
4. Run `EDA.py` to generate visual plots and inspect trends.

## Notes

- The repository was cleaned to avoid GitHub large-file restrictions.
- `Capstone/df_model.csv` is a very large file and may be excluded from the published repository.
- If you use Reddit scraping, configure PRAW credentials appropriately.
- This repo is intended for experimentation and may require additional cleanup before production use.

