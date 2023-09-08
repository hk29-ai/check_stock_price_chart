import random
import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from dateutil import relativedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib
plt.rcParams['font.size'] = 16 # グラフの基本フォントサイズの設定
import datetime as dt

# 東証上場銘柄一覧データを読み込む
df = pd.read_csv('data_j.csv')

# 銘柄をランダムに選択するため、乱数番号の取得
N = len(df)
random_No = str(random.randint(1, N))
print(f'{random_No}/{N}')

# 乱数番号より株銘柄データを抽出
data_df = df.iloc[int(random_No)]
data_df

# 証券コードと銘柄を取得
ticker_symbol = str(data_df['コード'])
bland = str(data_df['銘柄名'])

# 何年前からのデータを取得するか
delta_years = -4

# 今日の日付を取得する
now = dt.datetime.now()
st.write(now)

# 今日から数年前の日付を取得する（数か月前の場合は、monthsとする）
target_day = now + relativedelta.relativedelta(years = delta_years)

# webから時系列データを取得する
code = ticker_symbol + '.JP'
df = web.DataReader(code, 'stooq', target_day, now) # stooq, yahoo

df['Ave'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

# インデックスでソートする
DF = df.sort_index(ascending = True)

# 価格帯の最小と最大を算出
min = int(DF['Ave'].min())
max = int(DF['Ave'].max())

# 階級の境界を算出
my_classes = 21
my_bins = np.arange(min, max, int((max-min)/my_classes))

# 階級に分けて、カテゴリ列を作成する
DF['Category'] = pd.cut(DF['Ave'], my_bins)

#価格別出来高の計算
my_sum = DF.groupby('Category').sum()
#label_list = [str(i[0]) + 'a' + str(i[1]) for i in my_sum.index]
label_list = [str(i) for i in my_sum.index]

# 単純移動平均の計算
my_days1 = 25
my_days2 = 75
my_days3 = 200
simple_moving_average1 = pd.Series.rolling(DF['Close'], window=my_days1).mean()
simple_moving_average2 = pd.Series.rolling(DF['Close'], window=my_days2).mean()
simple_moving_average3 = pd.Series.rolling(DF['Close'], window=my_days3).mean()

# グラフ化
fig = plt.figure(figsize=(21,9))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(DF['Close'], color="k", lw=3)
ax1.set_ylabel('株価[￥]')
ax1.plot(simple_moving_average1, color="r", lw=3, label="移動平均 {} 日".format(my_days1))
ax1.plot(simple_moving_average2, color="y", lw=3, label="移動平均 {} 日".format(my_days2))
ax1.plot(simple_moving_average3, color="b", lw=3, label="移動平均 {} 日".format(my_days3))
ax1.legend()

# X軸目盛表記を調整する
x_ticklabels = ax1.get_xticklabels() # デフォルトの目盛り表記をゲットする
plt.setp(x_ticklabels, rotation=75) # 目盛り表記を90度回転。#フォントサイズの指定する場合 ,fontsize=16)
tick_spacing = 180 # 目盛り表示する間隔(3か月=90日)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) # X軸目盛の表示間隔を間引く
ax1.grid()
ax1.set_title(f'{ticker_symbol}, {bland}')

ax2 = fig.add_subplot(1, 2, 2)
ax2.barh(label_list, my_sum['Volume'], color="g")
ax2.set_xlabel('出来高')
ax2.set_ylabel('価格帯')
st.pyplot(fig)
