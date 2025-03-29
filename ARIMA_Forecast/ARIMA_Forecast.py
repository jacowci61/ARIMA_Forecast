# -*- coding: cp1251 -*-
from collections import Counter
from datetime import datetime
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import forecast
from pmdarima import auto_arima
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
# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
# salesValues = inputTable.drop(inputTable.columns[[0,1,-1]], axis = 1)
# salesValuesLIST =[]

# for i in range(salesValues.shape[0]):
#     salesValuesLIST.extend(salesValues.iloc[i].tolist())
# salesValuesDF = pd.DataFrame({'Кол-во продаж' : salesValuesLIST})

# dates = inputTable.columns[2:]
# date_start = pd.to_datetime(dates[0], dayfirst=True)
# date_end = pd.to_datetime(dates[-1], dayfirst=True)

# daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')

# date_range = pd.DataFrame({'Дата Продажи': daterng})

# # Extract unique region-product pairs
# unique_pairs = inputTable[['Регион продаж', 'Продукция']].drop_duplicates()

# transposition1 = unique_pairs.merge(date_range, how="cross")
# transposition1 = pd.concat([transposition1, salesValuesDF], axis = 1)
# transposition1.to_excel("combinations_test.xlsx")
# endregion

# region forecast
final_df = pd.read_excel('src2.xlsx')
# final_df = transposition1
grouped = final_df.groupby(['Регион продаж', 'Продукция'])

outputlist = pd.DataFrame(columns = ['Регион продаж', 'Продукция', 'Дата Продажи', 'Кол-во продаж'])

daterng = pd.date_range(start='2023-06-01', end='2025-01-01', freq='MS')

groupCounter = -1
for (region, product), group in grouped:
    groupCounter += 1
    forecast_dataframe = group[['Дата Продажи','Кол-во продаж']].copy()
    forecast_dataframe['Дата Продажи'] = pd.to_datetime(forecast_dataframe['Дата Продажи'])
    forecast_dataframe.set_index('Дата Продажи', inplace = True)
    p,d,q = auto_arima(pd.Series(forecast_dataframe['Кол-во продаж'].tolist(), index = daterng), seasonal = False, trace = True, stepwise = True).order

    model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'], order = (p,d,q))
    # outputlist.append(str(model.fit().forecast(steps = 1)).split(" "))
    # print(str(region), str(product), str(model.fit().forecast(steps = 1)).split(" ")[0], str(model.fit().forecast(steps = 1)).split(" ")[4].replace('\nFreq:', ''))
    outputlist.loc[groupCounter] = [str(region), str(product), str(model.fit().forecast(steps = 1)).split(" ")[0], str(model.fit().forecast(steps = 1)).split(" ")[4].replace('\nFreq:', '')]
outputlist.to_excel("output.xlsx")
# fix this ---------------
    # issue is in getting unique forecasted values to see if there are duplicates, because now amount of forecasted values and
    # amount of real values doesnt match
# outputtable = pd.DataFrame(outputlist)
# fix this ---------------
# print(outputtable)
# print(groupCounter)
# endregion