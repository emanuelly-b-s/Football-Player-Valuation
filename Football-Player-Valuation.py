# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:08:50 2022

@author: emanuelly-b-s
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#tratando os dados---------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("data.csv", header=0)

#Corrigindo os valores de mercado ($$$$$)
def numericValue(arr):
    dummy = []
    for i in range(len(arr)):
        if arr[i][len(arr[i])-1]=="M":
            dummy.append(pd.to_numeric(arr[i][0:len(arr[i])-2])*1000000)
        elif arr[i][len(arr[i])-1]=="K":
            dummy.append(pd.to_numeric(arr[i][0:len(arr[i])-2])*1000)
        else:
            dummy.append(0)
            
    return dummy

df["Value"]=df['Value'].str.replace('€','')
value = numericValue(df["Value"])
df["Value"]=np.array(value)/1000000


#deixando somente os goleiros
goleiros = df['Position']=="GK" 
dfiltrado = df[goleiros]


#goleiros dentro do orçamento
orçamento = dfiltrado["Value"]< 1
dfiltrado = dfiltrado[orçamento]

dados = dfiltrado[['GKDiving','GKHandling','GKKicking','GKReflexes','GKPositioning','GKReflexes','Value']]
dados = dados.dropna()





#separando amostras--------------------------------------------------------------------------------------------------------------------------

#To predict the "value" based on chosen attributes, defining y and x
y = dados['Value']
x = dados.drop(['Value'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.33, random_state=42) 


#instanciando e treinando os modelos---------------------------------------------------------------------------------------------------------

linreg = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()

linreg.fit(x_train, y_train)
dtr.fit(x_train, y_train)
rfr.fit(x_train, y_train)

y_pred_linear = linreg.predict(x_test)
y_pred_dtr = dtr.predict(x_test)
y_pred_rfr = rfr.predict(x_test)


# Avaliando os modelos----------------------------------------------------------------------------------------------------------------------
print('Linear Regression: ')
print('R2: {:.2f}'.format(r2_score(y_test, y_pred_linear)))
print('MAE: {:.2f}'.format(mean_absolute_error(y_test, y_pred_linear)))
print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_linear))))
print("Acurácia Linear Regression: {:.2f}%".format(linreg.score(x_test, y_test)*100))

print('\nDecision Tree Regressor: ')
print('R2: {:.2f}'.format(r2_score(y_test, y_pred_dtr)))
print('MAE: {:.2f}'.format(mean_absolute_error(y_test, y_pred_dtr)))
print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_dtr))))
print("Acurácia Decision Tree Regressor: {:.2f}%".format(dtr.score(x_test, y_test)*100))

print('\nRandom Forest Regressor: ')
print('R2: {:.2f}'.format(r2_score(y_test, y_pred_rfr)))
print('MAE: {:.2f}'.format(mean_absolute_error(y_test, y_pred_rfr)))
print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_rfr))))
print("Acurácia Random Forest Regressor: {:.2f}%".format(rfr.score(x_test, y_test)*100))

