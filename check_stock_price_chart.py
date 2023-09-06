import random
import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from dateutil import relativedelta
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import japanize_matplotlib
plt.rcParams['font.size'] = 16 # グラフの基本フォントサイズの設定


# 証券コード
#ticker_symbol = '3159'
while True:
    ticker_symbol = str(random.randint(1301, 9997))
    print(ticker_symbol)

    # 何年前からのデータを取得するか
    delta_years = -4

    # 今日の日付を取得する
    now = dt.datetime.now()
    # 今日から数年前の日付を取得する（数か月前の場合は、monthsとする）
    target_day = now + relativedelta.relativedelta(years = delta_years)

    # 時系列データを取得する
    code = ticker_symbol + '.JP'
    df = web.DataReader(code, 'stooq', target_day, now) # stooq, yahoo

    # インデックスでソートする
    DF = df.sort_index(ascending = True)
    
    # データフレームが空でない場合にループを抜ける
    if not DF.empty:
        print(DF)
        break
        
# データ抽出の区間を表示
print(target_day)
print(now)

# ATR計算の係数
atr_period = 10
atr_multiplier = 2

high = DF['High']
low = DF['Low']
close = DF['Close']

# ATR(Average True Range)の計算
price_diffs = [high - low, 
               high - close.shift(), 
               close.shift() - low]
true_range = pd.concat(price_diffs, axis=1)
true_range = true_range.abs().max(axis=1)
atr = true_range.ewm(alpha = 1 / atr_period, min_periods = atr_period).mean() 

# 上下バンドの計算
final_upperband = upperband = (high + low) / 2 + (atr_multiplier * atr)
final_lowerband = lowerband = (high + low) / 2 - (atr_multiplier * atr)
#print(final_upperband, final_lowerband)


# スーパートレンドを図示するための処理
# 上ラインは赤色、下ラインは緑で表記。さらに色塗りするため

supertrend = [True] * len(DF) # 一旦、Trueで埋める
for i in range(1, len(DF.index)):
    curr, prev = i, i-1
    
    if close[curr] > final_upperband[prev]:
        supertrend[curr] = True
    elif close[curr] < final_lowerband[prev]:
        supertrend[curr] = False
    # その他の場合は、既存トレンドを継続する
    else:
        supertrend[curr] = supertrend[prev]
        if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
            final_lowerband[curr] = final_lowerband[prev]
        if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
            final_upperband[curr] = final_upperband[prev]
    
    # トレンドでない方の列には、欠損値nanを代入（その区間は、図示させないための処理） 
    if supertrend[curr] == True:
        final_upperband[curr] = np.nan
    else:
        final_lowerband[curr] = np.nan

# 作成したデータをpandasデータフレームで作成する。        
df_buf = pd.DataFrame({
    'Supertrend': supertrend,
    'Final Lowerband': final_lowerband,
    'Final Upperband': final_upperband
}, index=DF.index)
# 元のデータフレームへ追記する
DF = DF.join(df_buf)
#print(DF)


# グラフ化
fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(111)
plt.plot(DF.index, DF['Close'], c = 'k', label='Close Price')
plt.plot(DF.index, DF['Final Lowerband'], c = 'lime', label = 'BUY')
plt.plot(DF.index, DF['Final Upperband'], c = 'red', label = 'SELL')

# 塗り潰し
ax.fill_between(DF.index, DF['Close'], DF['Final Lowerband'], facecolor='lime', alpha=0.3)
ax.fill_between(DF.index, DF['Close'], DF['Final Upperband'], facecolor='red', alpha=0.3)

# 横軸を時間フォーマットにする
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval = 1))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval = 3))
plt.gcf().autofmt_xdate()

plt.ylabel('stock price [\]')
plt.title(ticker_symbol)
plt.legend(
  bbox_to_anchor = (1.45, 0.85),
)
plt.grid()
fig.tight_layout()
#plt.show()
st.pyplot(fig)
