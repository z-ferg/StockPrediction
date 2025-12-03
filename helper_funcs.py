# Return the next trading day, avoiding holidays and weekends
#   Inputs
#       cur_day      -> The current date
#       trading_days -> List of all open dates in the range
#   Returns:
#       The next available date (datetime object)
def next_trading_day(cur_day, trading_days):
    days_left = trading_days[trading_days > cur_day]
    return days_left.min() if len(days_left) else trading_days.max()