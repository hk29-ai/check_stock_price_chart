import random
import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from dateutil import relativedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib
plt.rcParams['font.size'] = 24 # グラフの基本フォントサイズの設定
from prophet import Prophet
import datetime as dt
now = dt.datetime.now() # 今日の日付を取得する
st.write(now)

###################################################### データの読み込み
# 東証上場銘柄一覧データを読み込む
df = pd.read_csv('data_j.csv')

# 銘柄をランダムに選択するため、乱数番号の取得
N = len(df)
random_No = str(random.randint(1, N))

# 株銘柄データを抽出
data_df = df.iloc[int(random_No)]

st.write("""
### ■ランダムに選定された株の銘柄
""")
st.table(data_df)

# 証券コードと銘柄を取得
ticker_symbol = str(data_df['コード'])
bland = str(data_df['銘柄名'])

# 何年前からのデータを取得するか
delta_years = -4

# 今日から数年前の日付を取得する（数か月前の場合は、monthsとする）
target_day = now + relativedelta.relativedelta(years = delta_years)

# webから時系列データを取得する
code = ticker_symbol + '.JP'
df = web.DataReader(code, 'stooq', target_day, now) # stooq, yahoo

df['Ave'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

# インデックスでソートする
DF = df.sort_index(ascending = True)

##################################################### データ処理
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
label_list = [str(i) for i in my_sum.index]

# 単純移動平均の計算
my_days1 = 25
my_days2 = 75
my_days3 = 200
simple_moving_average1 = pd.Series.rolling(DF['Close'], window=my_days1).mean()
simple_moving_average2 = pd.Series.rolling(DF['Close'], window=my_days2).mean()
simple_moving_average3 = pd.Series.rolling(DF['Close'], window=my_days3).mean()

##################################################### Prophetによるモデルの作成
DF2 = DF.copy()
DF2['ds'] = DF2.index # 日付がインデックスにあるため、列にする
DF2 = DF2.rename(columns={'Close': 'y'}) # 目的変数の名前をyに置換する

# モデルの作成
model = Prophet(
    growth='linear', # 傾向変動の関数．非線形は'logistic'
    yearly_seasonality = True, # 年次の季節変動を考慮有無
    weekly_seasonality = False, # 週次の季節変動を考慮有無
    daily_seasonality = False, # 日次の季節変動を考慮有無
    changepoints = None, #  傾向変化点のリスト
    changepoint_range = 0.85, # 傾向変化点の候補の幅で先頭からの割合。
    changepoint_prior_scale = 0.5, # 傾向変化点の事前分布のスケール値。パラメータの柔軟性
    n_changepoints = 5, # 傾向変化点の数
) 
model.fit(DF2)

###################################################### グラフ化
fig = plt.figure(figsize=(21,9))

# 株価チャートのグラフを描く
ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])  # [左端, 下端, 幅, 高さ]
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
ax1.set_title(bland)

# 価格帯別の出来高のグラフを描く
ax2 = fig.add_axes([0.72, 0.1, 0.15, 0.8])
ax2.barh(label_list, my_sum['Volume'], color="g")
ax2.set_xlabel('出来高')
ax2.set_ylabel('価格帯')
st.write("""
### ■チャート情報と価格帯別の出来高
""")
st.pyplot(fig)


##### Prophetによる株価予測を図示
st.write("""
### ■Prophetによる株価予測
※右端の黒点がないあたりが予測結果です。
株価の予測についてはこれに限らず、鵜呑みにしないように注意ください。
""")

# 学習データに予測したい期間を追加する
future = model.make_future_dataframe(periods = 3, freq='M')

# 予測する
forecast = model.predict(future)

# プロット
fig2 = model.plot(forecast)
st.pyplot(fig2)
