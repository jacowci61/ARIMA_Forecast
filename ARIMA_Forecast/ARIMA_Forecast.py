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


def convert_to_datetime(df):
    for col in df.columns[2:]:  # пропускает первые два столбца (регион, название продукции), т.к они не являются датой
        try:
            if isinstance(col, str):  # если тип данных в столбце с датой является string
                df[col] = pd.to_datetime(col)
            
            elif df[col].dtype in ['int64', 'float64']:  # если тип данных в столбце с датой является int/float/любым типом данных который является числом
                df[col] = pd.to_datetime(df[col], unit='D')  # Assumes days since epoch
        except Exception as e:
            print(f"Conversion error for column {col}: {e}")
    return df

# region readingTable
# отвечает за получение данных с таблицы


# считывание файла можно сделать двумя путями:
# если файл НЕ находится в одной папке с файлом программы (.py), можно использовать абсолютный путь: inputTable = pd.read_excel('D:...\названиеПапки\названиеФайла.xlsx')
# если файл находится в одной папке с файлом программы (.py): inputTable = pd.read_excel('D:...\названиеПапки\названиеФайла.xlsx')
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')


salesValues = inputTable.drop(inputTable.columns[[0,1,-1,-2]], axis = 1)
# аргумент внутри метода .drop указывает на саму таблицу (inputTable) и индексы столбцов (inputTable.columns[[]]) которые будут удалены.
# цифры внутри .columns[[]] перечисляют индексы столбцов которые будут удалены. [[...,-1,-2]] удаляют столбцы с конца таблицы
salesValuesLIST =[]

for i in range(salesValues.shape[0]): # получение значений продаж
    salesValuesLIST.extend(salesValues.iloc[i].tolist())
salesValuesDF = pd.DataFrame({'Кол-во продаж' : salesValuesLIST})

dates = inputTable.columns[2:] # считывание диапазона дат со всех столбцов, кроме первых двух
date_start = pd.to_datetime(dates[0], dayfirst=True)
date_end = pd.to_datetime(dates[-1], dayfirst=True) # dates[-1] получает последний элемент с массива всех дат

daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')

date_range = pd.DataFrame({'Дата Продажи': daterng})

unique_pairs = inputTable[['Регион продаж', 'Продукция']].drop_duplicates() # получение уникальных комбинаций 'Регион продаж' + 'Продукция'

# следующие три строки отвечают за транспонирование для приведения таблицы из "wide" в "long" формат, который требуется для анализа временных рядов
transposition1 = unique_pairs.merge(date_range, how="cross")
transposition1 = pd.concat([transposition1, salesValuesDF], axis = 1)
transposition1.to_excel("combinations_test.xlsx")
# endregion

# region forecast
# отвечает за сам процесс прогнозирования


final_df = transposition1
grouped = final_df.groupby(['Регион продаж', 'Продукция']) # создание групп 'Регион продаж' + 'Продукция'
outputlist = pd.DataFrame(columns = ['Регион продаж', 'Продукция', 'Дата Продажи', 'Кол-во продаж'])
daterng = pd.date_range(start='2023-06-01', end='2024-12-01', freq='MS')

print(len(grouped))

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

    # ------------------
    diffirentiatedTS = originalTS
    # ------------------
    # print(originalTS)

    # if (originalTS.eq(0).all() == True):
    #     diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    #     continue

    # if ((adfuller(originalTS)[0] < adfuller(originalTS)[4]['1%']) == True): # raises warning if not enough non-zeroes to calculate
    #     isStationary = True
    #     diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    # else:
    #     isStationary = False
        
    # if (isStationary == False):
    #     while (isStationaryAfterDifferentiating == False):
    #         if (periodsAmount == 13):
    #             diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    #             break
    #         else:
    #             periodsAmount += 1
    #             try:
    #                 differenced = originalTS.diff(periods=periodsAmount).dropna()
    #                 # Check if the differenced series is constant
    #                 if differenced.nunique() <= 1:
    #                     # Skip this iteration if differenced series is constant
    #                     continue
                    
    #                 adf_result = adfuller(differenced)
    #                 if adf_result[0] < adf_result[4]['1%']:
    #                     isStationaryAfterDifferentiating = True
    #                     diffirentiatedTS = differenced
    #                     break
    #             except ValueError as e:
    #                 # Log error and continue with next period
    #                 print(f"Error with period {periodsAmount} for {region}-{product}: {e}")
    #                 continue

    maxP = len(diffirentiatedTS-1)
    maxQ = round(len(diffirentiatedTS) / 10)
    best_P_Q = []

    print()
    print()
    print()

    print(groupCounter)

    print()
    print()
    print()

    for p in range(0,maxP):
        for q in range(0, maxQ):
            if (q != 0):
                fittedModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q))
                fittedModelQ1.initialize_approximate_diffuse()
                fittedModelQ = fittedModelQ1.fit()
                if (fittedModelQ.aic < bestModelQ.aic):
                    bestModelQ = fittedModelQ
            else:
                if (maxQ >= 2):
                    model3 = sm.tsa.arima.ARIMA # buffer for best model if Q >=2. Not the case for current range, but it should be dynamic anyway
                    bestModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q))
                    bestModelQ1.initialize_approximate_diffuse()
                    bestModelQ = bestModelQ1.fit()
                else:
                    bestModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q))
                    bestModelQ1.initialize_approximate_diffuse()
                    bestModelQ = bestModelQ1.fit()
        if (p != 0):
          fittedModelP1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,bestModelQ.model_orders['ma']))
          fittedModelP1.initialize_approximate_diffuse()
          fittedModelP = fittedModelP1.fit()
          if (fittedModelP.aic < bestModelP.aic):
                bestModelP = fittedModelP
        else:
          bestModelP = bestModelQ

    forecastValue = bestModelP.forecast(steps=1).iloc[0]
    forecastDate = bestModelP.forecast(steps=1).index[0]
    outputlist.loc[groupCounter] = [str(region), str(product), forecastDate, forecastValue]
outputlist.to_excel("output.xlsx")
# endregion