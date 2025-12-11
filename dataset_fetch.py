import os
import zipfile
import pandas as pd


def fetch_datasets(news_zip_path, stock_zip_path, news_extract_path, stock_extract_path):
    """ Function to extract datasets from zip files

    args:
        news_zip_path      -> Path to the news zip file
        stock_zip_path     -> Path to the stock zip file
        news_extract_path  -> Directory to extract news data
        stock_extract_path -> Directory to extract stock data
    
    rets:
        None
    """

    # Create extraction directories if they do not exist
    news_parent = os.path.dirname(news_extract_path) or '.'
    stock_parent = os.path.dirname(stock_extract_path) or '.'

    # Extract news data
    if os.path.exists(news_extract_path) and len(os.listdir(news_extract_path)) > 0:
        print(f"News data already extracted in {news_extract_path}. Skipping extraction.")
    else:
        try:
            with zipfile.ZipFile(news_zip_path, 'r') as zip_ref:
                zip_ref.extractall(news_parent)
            print(f"Contents of {news_zip_path} extracted to {news_extract_path} successfully.")
        except zipfile.BadZipFile as e:
            print(f"Error: {e}. The file may be corrupted or not a zip file.")
        except FileNotFoundError as e:
            print(f"Error: {e}. The specified file was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    # Extract stock data
    if os.path.exists(stock_extract_path) and len(os.listdir(stock_extract_path)) > 0:
        print(f"Stock data already extracted in {stock_extract_path}. Skipping extraction.")
    else:
        try:
            with zipfile.ZipFile(stock_zip_path, 'r') as zip_ref:
                zip_ref.extractall(stock_parent)
            print(f"Contents of {stock_zip_path} extracted to {stock_extract_path} successfully.")
        except zipfile.BadZipFile as e:
            print(f"Error: {e}. The file may be corrupted or not a zip file.")
        except FileNotFoundError as e:
            print(f"Error: {e}. The specified file was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def fetch_stock_datasets(stock_meta_file, stock_data_path):
    """ Function for fetching individual stock datasets based on metafile

    args:
        stock_meta_file  -> DataFrame containing stock metadata
        stock_data_path  -> Path to the directory containing stock CSV files
    
    rets:
        stock_dataframes -> Dictionary of DataFrames for each stock ticker
    """
    tickers = [x['Symbol'] for _, x in stock_meta_file.iterrows() if x['ETF'] == 'N'] # Exclude ETFs
    stock_dataframes = {}

    for t in tickers:
        try:
            stock_dataframes[t] = pd.read_csv(f"{stock_data_path}/stocks/{t}.csv")
        except Exception:
            print(f"Could not load data for ticker: {t}")
    
    return stock_dataframes