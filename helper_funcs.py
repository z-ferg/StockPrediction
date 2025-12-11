import zipfile
import os
import numpy as np

def next_trading_day(cur_day, trading_days):
    """ Application function to get the next trading day from current day.

        args:
            cur_day       -> Current date (pd.Timestamp)
            trading_days  -> Series of trading days (pd.Series of pd.Timestamp)
        
        rets:
            next_day      -> Next trading day (pd.Timestamp)
    """
    days_left = trading_days[trading_days > cur_day]
    return days_left.min() if len(days_left) else trading_days.max()

def mda(y_true, preds):
    """ Custom Mean Directional Accuracy metric for evaluating model predictions

        args:
            y_true  -> True target values (pd.Series or np.array)
            preds   -> Predicted target values (pd.Series or np.array)
        
        rets:
            MDA     -> Mean Directional Accuracy (float)
    """
    return float(np.mean(np.sign(preds) == np.sign(y_true)))