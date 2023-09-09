import random
import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib
plt.rcParams['font.size'] = 24 # グラフの基本フォントサイズの設定

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers

import datetime as dt
from dateutil import relativedelta
from datetime import timedelta

now = dt.datetime.now() # 今日の日付を取得する
st.write(now)

###################################################### 1. 株価データの読み込み
# 東証上場銘柄一覧データを読み込む
df = pd.read_csv('data_j.csv')

# 銘柄をランダムに選択するため、乱数番号の取得
N = len(df)
random_No = str(random.randint(1, N))

# 株銘柄データを抽出
data_df = df.iloc[int(random_No)]

st.write("""
### ■1. ランダムに選択した株の銘柄
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

##################################################### 2. 株価チャート（移動平均の計算, 価格帯別の出来高の計算）
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

### データの可視化
fig = plt.figure(figsize=(21,9))

# 株価チャートのグラフを描く
ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])  # [左端, 下端, 幅, 高さ]
ax1.plot(DF['Close'], color="k", lw=3)
ax1.set_ylabel('株価 [円]')
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
ax1.set_title(f'{bland}：{ticker_symbol}')

# 価格帯別の出来高のグラフを描く
ax2 = fig.add_axes([0.72, 0.1, 0.15, 0.8])
ax2.barh(label_list, my_sum['Volume'], color="g")
ax2.set_xlabel('出来高')
ax2.set_ylabel('価格帯')
st.write("""
### ■2. チャートと価格帯別の出来高
下図左は株価チャートで、移動平均線を計算してプロットしています。  下図右は出来高を価格帯別に算出して図示しています。
""")
st.pyplot(fig)


##################################################### 3. Prophetによる予測
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

#### データの可視化
st.write("""
### ■3. Prophetを用いた株価予測
時系列データ分析のPythonライブラリ「Prophet」による株価予測です。下図中の右端の黒点がないあたりが予測結果です。
""")

# 学習データに予測したい期間を追加する
future = model.make_future_dataframe(periods = 3, freq='M')

# 予測する
forecast = model.predict(future)

### グラフの作成
plt.rcParams['font.size'] = 20 # グラフの基本フォントサイズの設定
fig, ax = plt.subplots(figsize=(10,6))

# グラフオブジェクトを取得
fig = model.plot(forecast, ax=ax)

# x軸の目盛間隔を分割
ax.locator_params(axis='x', nbins=30)
# x軸目盛を回転
ax.tick_params(axis='x', rotation=60)

# タイトル、ラベルの設定
ax.set_title(f'{bland}：{ticker_symbol}')
ax.set_xlabel('')
ax.set_ylabel('株価 [円]')

st.pyplot(fig)

##################################################### RNNによる予測
df_close = DF["Close"] # 終値のみ使用

# データの正規化
max_value = df_close.max()
min_value = df_close.min()
df_close_normalize = (df_close - min_value) / (max_value - min_value)

# 学習用とテスト用に分割
train_size = int(len(df_close_normalize) * 0.75) # 学習用データの割合
train = df_close_normalize[:train_size]
test = df_close_normalize[train_size:]

# 入出力を作成する関数
def create_dataset(_data, _window_size):
    X, y = [], []
    for i in range(len(_data) - _window_size):
        X.append(_data[i:i + _window_size])
        y.append(_data[i + _window_size])
    return np.array(X), np.array(y)

window_size = 14 # 学習データの日にち区間
X_train, y_train = create_dataset(train, window_size)
X_test, y_test = create_dataset(test, window_size)

# LSTMの入力形式へ変換
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# モデルの構築
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50, activation='relu', return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# オプティマイザーの設定
my_optimizer = optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
)

# モデルのコンパイル
model.compile(optimizer = my_optimizer, loss = "mean_squared_error")

# モデルの学習
model.fit(X_train, y_train, epochs=10, batch_size=32) # エポック数, バッチサイズ

# モデルの評価
train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
st.write("Train Score: {:.4f}".format(train_score))
st.write("Test Score: {:.4f}".format(test_score))

# 日付データの取得
dates = df_close_normalize.index # 日付データがインデックスのため、取得
dates = pd.to_datetime(dates) # 日付データをdatetime型に変換

# 訓練用データの日付データの取得
train_dates = dates[:train_size]
# テスト用データの日付データの取得
test_dates = dates[train_size + window_size:] 

# 予測値の取得
y_pred = model.predict(X_test)
# 予測値の逆正規化
y_pred = y_pred * (max_value - min_value) + min_value

# 未来の1か月分の日付データの作成
pred_days = 90 # 予測したい日数

last_date = test_dates[-1] # 最後の日付を取得
future_dates = [last_date + timedelta(days=i) for i in range(1, pred_days + 1)] # 最後の日付から1日ずつ増やしてリストに追加
future_dates = pd.to_datetime(future_dates) # 日付データをdatetime型に変換

# 未来の予測値の取得
future_pred = []
last_input = X_test[-1] # 最後の入力データを取得

# 予測したい日数分実行する
for i in range(pred_days):
    pred = model.predict(last_input.reshape(1, window_size, 1)) # 入力データから予測値を取得
    future_pred.append(pred[0][0]) # 予測値をリストに追加
    last_input = np.append(last_input[1:], pred) # 入力データを更新

# 未来の予測値の逆正規化
future_pred = np.array(future_pred) * (max_value - min_value) + min_value

### データの可視化
st.write("""
### ■4. RNNによる株価予測
Pythonライブラリ「keras」を用いて、RNNの1種であるLSTM（Long Short-Term Memory）による株価予測です。下図中の右端のピンクが未来予測です。
""")
fig, ax = plt.subplots(figsize=(10,6))
# 訓練データ
plt.plot(train_dates, train * (max_value - min_value) + min_value, label="Train")
# テストデータ
plt.plot(test_dates, test[window_size:] * (max_value - min_value) + min_value, label="Actual")
# テストデータ部分の予測
plt.plot(test_dates, y_pred, label="Predicted")
# 未来の予測
plt.scatter(future_dates, future_pred, label="Future", marker='o', linewidth=3, color='magenta')

plt.title(f'{bland}：{ticker_symbol}')
plt.ylabel('株価 [円]')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

st.write("""
※一般的に株価の予測については、これらに限らず鵜呑みにしないように注意ください。
結局、株価の変動は外的要因とそれによって生じる心理的要因が大きいと思います
""")
