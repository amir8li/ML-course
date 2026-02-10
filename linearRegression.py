import pandas as pd
import yfinance as yf

df = yf.download('GOOGL')
print(df.head())

