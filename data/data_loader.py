import yfinance as yf
from cache.model_cache import global_stock_data

def get_stock_data(ticker="AAPL", period="1y"):

    if ticker in global_stock_data:
        return global_stock_data[ticker]

    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)

    global_stock_data[ticker] = df[['Close']]

    return global_stock_data[ticker]