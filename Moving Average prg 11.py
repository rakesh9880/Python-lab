#11.Write a python program to implement Forecasting using Moving Average.
import pandas as pd

from movingaverage import window_size

data_path="C:/Users/HP/Downloads/daily-min-temperatures.csv"
df=pd.read_csv(data_path,parse_dates=['Date'],index_col='Date')

series=df['Temp']

window_size=7
movingaverage=series.rolling(window=window_size).mean()

forecast=movingaverage.iloc[-1]

print(f"Forecasted next value using{window_size}-day Moving average:{forecast:.3f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(series,label="Daily Min Temparature")
plt.plot(movingaverage,label=f'{window_size}-Daily Moving Average',linewidth=2)
plt.xlabel('Date')
plt.ylabel('Temparature("c")')
plt.title('Moving Average Forecasting on Daily Min Temparature(melbourne)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
