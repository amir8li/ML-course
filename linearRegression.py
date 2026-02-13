import pandas as pd
import yfinance as yf
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

def get_quandl_style_data(ticker, start_date='2020-01-01', end_date='2023-12-31'):
    t = yf.Ticker(ticker)
    df = t.history(start=start_date, end=end_date)
    
    
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
        
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = df['Close'] if 'Close' in df.columns else 0
    
    adj_factor = df['Adj Close'] / df['Close']
    
    result = pd.DataFrame(index=df.index)
    result['Adj. Open'] = df['Open'] * adj_factor
    result['Adj. High'] = df['High'] * adj_factor
    result['Adj. Low'] = df['Low'] * adj_factor
    result['Adj. Close'] = df['Adj Close']
    result['Adj. Volume'] = df['Volume'] / adj_factor
    
    return result

df = get_quandl_style_data('GOOGL')

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

for_cast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[for_cast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f'Accuracy: {accuracy}')