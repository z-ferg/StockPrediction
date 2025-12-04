import os
import pandas as pd
import helper_funcs as hf

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
            print(f"Error with stock: {s}")
    
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

        if df.shape[0] != 2057: # 2057 dates 2012-01-28 and 2020-04-01, cut unfilled stocks
            continue

        df['r_0d'] = (df['Close'] - df['Open'])/df['Open']
        df['r_1d'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
        df['r_7d'] = (df['Close'].shift(-7) - df['Close']) / df['Close']
        df['r_30d'] = (df['Close'].shift(-30) - df['Close']) / df['Close']

        processed_stocks[s] = df

    print(f"Number of Stocks before processing: \t{len(stock_dfs)}")
    print(f"Number of Stocks after processing: \t{len(processed_stocks)}\n")

    return {k: v.dropna().reset_index(drop=True) for k, v in processed_stocks.items()}


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
    return news_df