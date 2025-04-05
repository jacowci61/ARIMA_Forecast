# -*- coding: cp1251 -*-
from collections import Counter
from datetime import datetime
from itertools import combinations
from _pytest.pathlib import bestrelpath
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.vector_ar.vecm import forecast
from pmdarima import auto_arima
from matplotlib import pyplot
import string
import pandas as pd
import numpy as np
import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA


def convert_to_datetime(df):
    for col in df.columns[2:]:  # Skip first two columns
        try:
            # If column is string representation of date in column name
            if isinstance(col, str):
                df[col] = pd.to_datetime(col)
            
            # If column contains numeric values
            elif df[col].dtype in ['int64', 'float64']:
                df[col] = pd.to_datetime(df[col], unit='D')  # Assumes days since epoch
        except Exception as e:
            print(f"Conversion error for column {col}: {e}")
    return df

# region readingTable
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
salesValues = inputTable.drop(inputTable.columns[[0,1,-1,-2]], axis = 1)
salesValuesLIST =[]

for i in range(salesValues.shape[0]):
    salesValuesLIST.extend(salesValues.iloc[i].tolist())
salesValuesDF = pd.DataFrame({'Кол-во продаж' : salesValuesLIST})

dates = inputTable.columns[2:]
date_start = pd.to_datetime(dates[0], dayfirst=True)
date_end = pd.to_datetime(dates[-1], dayfirst=True)

daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')

date_range = pd.DataFrame({'Дата Продажи': daterng})

# Extract unique region-product pairs
unique_pairs = inputTable[['Регион продаж', 'Продукция']].drop_duplicates()

transposition1 = unique_pairs.merge(date_range, how="cross")
transposition1 = pd.concat([transposition1, salesValuesDF], axis = 1)
# transposition1.to_excel("combinations_test.xlsx")
# endregion

# region forecast
# final_df = pd.read_excel('src2.xlsx')
final_df = transposition1
grouped = final_df.groupby(['Регион продаж', 'Продукция'])
outputlist = pd.DataFrame(columns = ['Регион продаж', 'Продукция', 'Дата Продажи', 'Кол-во продаж'])
daterng = pd.date_range(start='2023-06-01', end='2024-12-01', freq='MS')

isStationary = bool
isStationaryAfterDifferentiating = bool
periodsAmount = 0
groupCounter = -1

for (region, product), group in grouped:
    groupCounter += 1
    periodsAmount = 0
    currentP = 0
    isStationary = False
    isStationaryAfterDifferentiating = False
    forecast_dataframe = group[['Дата Продажи','Кол-во продаж']].copy()
    forecast_dataframe['Дата Продажи'] = pd.to_datetime(forecast_dataframe['Дата Продажи'])
    forecast_dataframe.set_index('Дата Продажи', inplace = True)
    originalTS =pd.Series(forecast_dataframe['Кол-во продаж'].tolist(), index = daterng)
    diffirentiatedTS = pd.Series

    if (originalTS.eq(0).all() == True):
        diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
        continue

    if ((adfuller(originalTS)[0] < adfuller(originalTS)[4]['1%']) == True): # raises warning if not enough non-zeroes to calculate
        isStationary = True
        diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    else:
        isStationary = False
        
    if (isStationary == False):
        while (isStationaryAfterDifferentiating == False):
            if (periodsAmount == 13):
                diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
                break
            else:
                periodsAmount += 1
                if ((adfuller(originalTS.diff(periods = periodsAmount).dropna())[0] < adfuller(originalTS.diff(periods = periodsAmount).dropna())[4]['1%']) == True):
                    isStationaryAfterDifferentiating = True
                    diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
                    break

    maxP = len(diffirentiatedTS-1)
    maxQ = round(len(diffirentiatedTS) / 10)
    best_P_Q = []

    for p in range(0,maxP):
        for q in range(0, maxQ):
            if (q != 0):
                fittedModelQ = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q)).fit()
                if (fittedModelQ.aic < bestModelQ.aic):
                    bestModelQ = fittedModelQ
            else:
                if (maxQ >= 2):
                    model3 = sm.tsa.arima.ARIMA # buffer for best model if Q >=2. Not the case for current range, but it should be dynamic anyway
                    bestModelQ = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q)).fit()
                else:
                    bestModelQ = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q)).fit()
        if (p != 0):
          fittedModelP = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,bestModelQ.model_orders['ma'])).fit()
          if (fittedModelP.aic < bestModelP.aic):
                bestModelP = fittedModelP
        else:
          bestModelP = bestModelQ

    forecastValue = bestModelP.forecast(steps=1).iloc[0]
    forecastDate = bestModelP.forecast(steps=1).index[0]
    outputlist.loc[groupCounter] = [str(region), str(product), forecastDate, forecastValue]
outputlist.to_excel("output.xlsx")
# endregion