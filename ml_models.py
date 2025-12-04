from textblob import TextBlob
import pandas as pd


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