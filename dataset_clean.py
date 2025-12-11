import os
import pandas as pd
import helper_funcs as hf


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
    news_df['effective_date'] = news_df['date'].apply(lambda d: hf.next_trading_day(d, trading_days))

    # Stock ticker named TECH has issues in a later step
    news_df['category'] = news_df['category'].replace({'TECH': 'TECHNOLOGY'})

    # Capitalize date to fit with column name in stock_data for merging
    news_df = news_df.rename(columns={"date": "Date"})

    return news_df


#==================================================================#
#                      Dataset Concatenation
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
        sent_df = daily_sentiments[daily_sentiments['Date'] == d]
        daily_sent_scores = [sent_df[sent_df['category'] == cat]['avg_sentiment'].item() for cat in all_cats]
        
        # Iterate over tickers
        for ticker, df in stock_dfs.items():
            date, open, close, tick, r0, r1, r7, r30 = df[df['Date'] == d].values[0]
            
            row = [d, tick, open, r0, r1, r7, r30] # Get the date and stock ticker
            row.extend(daily_sent_scores)
            
            all_rows.append(row)

    return pd.DataFrame(data=all_rows, columns=cols)