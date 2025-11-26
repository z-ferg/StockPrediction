import os, sys
import pandas as pd

if __name__ == '__main__':
    """ Local Dataset Importing
            Originally tried to use the Kaggle API however quickly faced
            issues with rate limiting. Local downloading will speed up
            training at the cost of repeatability for others viewing project.
            
            Solution, I pull from downloaded datasets in parent folder to this
            github repository. news_data contains only the news dataset .json file,
            stock data contains a .csv with the metadata as well as 2 subfolders
            ETF and Stocks that contain their respective ticker histories """
    
    news_path = os.getcwd() + "/../news_data/"
    stock_path = os.getcwd() + "/../stock_data/"
    
    news_df = pd.read_json(f'{news_path}News_Category_Dataset_v3.json', lines=True)
    stock_meta = pd.read_csv(f'{stock_path}symbols_valid_meta.csv')