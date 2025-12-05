import os
import pandas as pd

from textblob import TextBlob
import pandas as pd


# Imports
import sys

import pandas as pd
from textblob import TextBlob

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#==================================================================#
#                   Dataset Fetching Functions
#==================================================================#

"""
    Small helper function to fetch stock meta file and full news dataframe
        ** Should be noted that files are far too large to place on Github
        ** so files are kept local on machine, more information in .readme
        
    args -> relative path to news data, relative path to stock data
    rets -> full news dataframe, meta file for stock data
"""
def fetch_datasets(rel_news_path, rel_stock_path):
    abs_news_path = os.getcwd() + rel_news_path
    abs_stock_path = os.getcwd() + rel_stock_path
    
    news_df = pd.read_json(f'{abs_news_path}News_Category_Dataset_v3.json', lines=True)
    stock_meta = pd.read_csv(f'{abs_stock_path}symbols_valid_meta.csv')
    
    return news_df, stock_meta

"""
    Function to retrieve all stock dataframes.
    
    args -> stock meta containing all stock information, relative path to stock dataframes
    rets -> dictionary with keys as tickers and values as stock dataframes
"""
def fetch_stock_data(stock_meta, rel_stock_path):
    stock_tickers = [x['Symbol'] for _, x in stock_meta.iterrows() if x['ETF'] == 'N'] # Remove ETFs
    stock_dfs = {}
    
    abs_stock_path = os.getcwd() + rel_stock_path

    for s in stock_tickers:
        try:
            stock_dfs[s] = pd.read_csv(f"{abs_stock_path}/stocks/{s}.csv")
        except Exception:
            print(f"Error retrieving stock: {s}")
    
    #Just formatting for cleaner output clarity
    print()
    
    return stock_dfs


#==================================================================#
#                   Dataset Cleaning Functions
#==================================================================#

"""
    Function for cleaning and organizing stock information.
        Drop irrelevant columns --> High, Low, Volume, adj_close
        Drop information prior to 2012-01-28
        Create stock return time horizon features
    
    args -> list of raw stock dataframes
    rets -> list of cleaned stock dataframes
"""
def prep_stock_data(stock_dfs):
    processed_stocks = {}
    tickers = stock_dfs.keys()

    for s in tickers:
        df = stock_dfs[s].copy()

        df['Ticker'] = s
        df['Date'] = pd.to_datetime(df['Date'])
        
        df = df[df['Date'] >= pd.Timestamp("2012-01-28")].reset_index(drop=True)
        df = df.drop(['High', 'Low', 'Adj Close', 'Volume'], axis=1)

        # 2057 dates 2012-01-28 and 2020-04-01, cut unfilled stocks
        if df.shape[0] != 2057:
            continue
        
        # Remove invalid or volatile stocks (0, negative, penny)
        df = df[(df['Open'] > 0.5) & (df['Close'] > 0.5)]
        
        if df.shape[0] < 2057:
            continue
        
        df['r_0d'] = (df['Close'] - df['Open']) / df['Open']
        df['r_1d'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
        df['r_7d'] = (df['Close'].shift(-7) - df['Close']) / df['Close']
        df['r_30d'] = (df['Close'].shift(-30) - df['Close']) / df['Close']
        
        if df['r_1d'].abs().max() > 100:
            print(f"Dropping {s}: invalid 1 day return")
            continue
        
        df = df.dropna().reset_index(drop=True)
        
        processed_stocks[s] = df

    print(f"\nNumber of Stocks before processing: \t{len(stock_dfs)}")
    print(f"Number of Stocks after processing: \t{len(processed_stocks)}\n")

    return processed_stocks


"""
    Function to clean and prep news dataframe for sentiment analysis.
        News DF starts 2022-09-23, ends 2012-01-28 --> First must reverse dataset
        Drop categories to only necessary --> Category, Headline, Date
        Shift dates to align with trading days, skipping weekends and holidays till next open day
    
    args -> raw news dataframe and dictionary of stock dataframes
    rets -> cleaned dataframe containing news content
"""
def prep_news_data(news_df, stock_dfs):
    news_df = news_df.sort_values(by='date', ascending=True).reset_index(drop=True)
    news_df = news_df.drop(['link', 'short_description', 'authors'], axis=1)
    news_df = news_df[news_df['date'] <= pd.Timestamp("2020-04-01")].reset_index(drop=True)

    trading_days = pd.to_datetime(stock_dfs['A']['Date'].unique())
    news_df['effective_date'] = news_df['date'].apply(lambda d: next_trading_day(d, trading_days))

    # Stock ticker named TECH has issues in a later step
    news_df['category'] = news_df['category'].replace({'TECH': 'TECHNOLOGY'})
    return news_df


# Return the next trading day, avoiding holidays and weekends
#   Inputs
#       cur_day      -> The current date
#       trading_days -> List of all open dates in the range
#   Returns:
#       The next available date (datetime object)
def next_trading_day(cur_day, trading_days):
    days_left = trading_days[trading_days > cur_day]
    return days_left.min() if len(days_left) else trading_days.max()


#==================================================================#
#                   Sentiment Analysis Modeling
#==================================================================#
"""

"""
def sentiment_analysis(news_df):
    # Create sentiment of each headline using 
    news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

    daily_sentiment = (
        news_df.groupby(['effective_date', 'category'])
        .agg(
            avg_sentiment = ('sentiment', 'mean'),
            article_count = ('headline', 'count')
        )
        .reset_index()
    )
    
    all_dates = pd.date_range(news_df['effective_date'].min(), news_df['effective_date'].max())
    all_cats = news_df['category'].unique()


    all_cats_per_date = pd.MultiIndex.from_product(
        [all_dates, all_cats],
        names=['date', 'category']
    )

    daily_sentiment = (
        daily_sentiment
        .set_index(['effective_date', 'category'])
        .reindex(all_cats_per_date)
        .fillna({'avg_sentiment': 0, 'article_count': 0})
        .reset_index()
    )
    
    return daily_sentiment


#==================================================================#
#                       XGBoost Modeling
#==================================================================#
"""

"""
def dataframe_union(stock_dfs, daily_sentiments):
    """
        At this point:
            - stock_dfs: dictionary with ticker as key, DF as value
                - Gather list of tickers being used through stock_dfs.keys()
                - stock_dfs[...]
                    - Date, Open, Close, Ticker, r_0d, r_1d, r_7d, r_30d
            - daily_sentiment: DF holding sentiment scores
                - date, category (42 total), avg_sentiment, article_count
        
        The plan:
            - For each time horizon create an XGBoost Model
                - Each model gets all 3511 stock openings and the scores of that day
                    - Target is the designated time horizon of that modelSo 
    """
    all_rows = []
    
    all_dates = stock_dfs['A']['Date']
    all_cats = daily_sentiments['category'].unique()

    cols = ['Date', 'Ticker', 'Open', 'r_0d', 'r_1d', 'r_7d', 'r_30d']
    cols.extend(all_cats)

    for d in all_dates:
        # Get dictionary of daily sentiments
        sent_df = daily_sentiments[daily_sentiments['date'] == d]
        daily_sent_scores = [sent_df[sent_df['category'] == cat]['avg_sentiment'].item() for cat in all_cats]
        
        # Iterate over tickers
        for ticker, df in stock_dfs.items():
            if ticker == "ALRS":
                print(d)
                print(df)
                
            date, open, close, tick, r0, r1, r7, r30 = df[df['Date'] == d].values[0]
            
            row = [d, tick, open, r0, r1, r7, r30] # Get the date and stock ticker
            row.extend(daily_sent_scores)
            
            all_rows.append(row)

    return pd.DataFrame(data=all_rows, columns=cols)


if __name__ == "__main__":
    # Fetch all datasets
    news_path = "/../../news_data/"
    stock_path = "/../../stock_data/"

    news_df, stock_meta = fetch_datasets(news_path, stock_path)
    stock_dfs = fetch_stock_data(stock_meta, stock_path)

    # Prepare datasets
    stock_dfs = prep_stock_data(stock_dfs)
    news_df = prep_news_data(news_df, stock_dfs)

    # Sentiment Analysis
    daily_sent = sentiment_analysis(news_df)
    
    big_df = dataframe_union(stock_dfs, daily_sent)
    
    # Experiment with 30 day return
    x_features = ['Open'] + list(news_df['category'].unique())

    X = big_df[x_features]
    y_30d = big_df['r_30d']

    X_train, X_test, y_train, y_test = train_test_split(X, y_30d, test_size=0.2, random_state=42)

    N_EST = 1000
    LAMBDA = 0.05
    MAX_DEPTH = 6
    SUB_SAMP = 0.8
    OBJECTIVE = "reg:squarederror"
    METHOD = "hist"

    model_30d = xgb.XGBRegressor(
        n_estimators = N_EST,
        learning_rate = LAMBDA,
        max_depth = MAX_DEPTH,
        subsample = SUB_SAMP,
        objective = OBJECTIVE,
        tree_method = METHOD
    )

    model_30d.fit(
        X_train, 
        y_train,
        eval_set=[(X_test,y_test)],
        verbose=50
    )

    predictions = model_30d.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"RMSE: {rmse:,.6f}")
    print(f"MAE:  {mae:,.6f}")
    print(f"R²:   {r2:,.6f}")