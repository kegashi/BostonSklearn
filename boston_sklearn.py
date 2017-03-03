#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor as Regressor
#from sklearn.ensemble import BaggingRegressor as Regressor
#from sklearn.linear_model import LinearRegressor as Regressor
#from sklearn.linear_model import Ridge as Regressor
#from sklearn.linear_model import Lasso as Regressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#ボストンのデータセットを取得
boston = load_boston()
df = DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = np.array(boston.target)

#説明変数XにMEDV以外の全ての要素, yにMEDV
X = df.iloc[:, :-1].values
y = df.loc[:, 'MEDV'].values

#学習用とテスト用に分割
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = 666)

#学習
forest = Regressor()
forest.fit(X_train, y_train)

#学習器が予測した結果を計算する
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

#MSE(平均二条誤差)を計算する
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

#R^2の計算
print('R^2 : %.3f, test : %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
