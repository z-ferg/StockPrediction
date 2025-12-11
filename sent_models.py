from textblob import TextBlob
import pandas as pd


#==================================================================#
#                   Sentiment Analysis Modeling
#==================================================================#
def sentiment_analysis(news_df):
    """ Generate sentiment analysis on news headlines.
            Apply TextBlob sentiment analysis on each headline.
            Aggregate daily average sentiment scores by category.

        args:
            news_df       -> Cleaned news dataframe (pd.DataFrame)
        
        rets:
            Sentiment dataframe with daily average sentiment scores by category (pd.DataFrame)
    """
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
        names=['Date', 'category']
    )

    daily_sentiment = (
        daily_sentiment
        .set_index(['effective_date', 'category'])
        .reindex(all_cats_per_date)
        .fillna({'avg_sentiment': 0, 'article_count': 0})
        .reset_index()
    )
    
    return daily_sentiment