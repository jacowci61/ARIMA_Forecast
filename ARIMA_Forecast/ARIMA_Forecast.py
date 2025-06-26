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
                df[col] = pd.to_datetime(df[col], unit='D')  # assumes days since epoch
        except Exception as e:
            print(f"Conversion error for column {col}: {e}")
    return df

# region readingTable
# отвечает за получение данных с таблицы


# считывание файла можно сделать двумя путями:
# если файл НЕ находится в одной папке с файлом программы (.py), можно использовать абсолютный путь: inputTable = pd.read_excel('D:...\названиеПапки\названиеФайла.xlsx')
# если файл находится в одной папке с файлом программы (.py): inputTable = pd.read_excel('D:...\названиеПапки\названиеФайла.xlsx')
inputTable = pd.read_excel('D:\Прогнозирование\src.xlsx')  # пример пути, всегда должен изменяться если файл НЕ находится в одной папке с файлом программы (.py). Путь к файлу можно получить через *ПКМ по файлу*->свойства->вкладка "безопасность"->первая строка "имя объекта"


salesValues = inputTable.drop(inputTable.columns[[0,1,-1,-2]], axis = 1)
# аргумент внутри метода .drop указывает на саму таблицу (inputTable) и индексы столбцов (inputTable.columns[[]]) которые будут удалены.
# цифры внутри .columns[[]] перечисляют индексы столбцов которые будут удалены. [[...,-1,-2]] удаляют столбцы с конца таблицы
salesValuesLIST =[]

for i in range(salesValues.shape[0]):  # получение значений продаж
    salesValuesLIST.extend(salesValues.iloc[i].tolist())
salesValuesDF = pd.DataFrame({'Кол-во продаж' : salesValuesLIST})

dates = inputTable.columns[2:]  # считывание диапазона дат со всех столбцов, кроме первых двух
date_start = pd.to_datetime(dates[0], dayfirst=True)
date_end = pd.to_datetime(dates[-1], dayfirst=True)  # dates[-1] получает последний элемент с массива всех дат

print(salesValues.columns[0])
daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')

date_range = pd.DataFrame({'Дата Продажи': daterng})

unique_pairs = inputTable[['Регион продаж', 'Продукция']].drop_duplicates()  # получение уникальных комбинаций 'Регион продаж' + 'Продукция'

# следующие три строки отвечают за транспонирование для приведения таблицы из "wide" в "long" формат, который требуется для анализа временных рядов
transposition1 = unique_pairs.merge(date_range, how="cross")
transposition1 = pd.concat([transposition1, salesValuesDF], axis = 1)
# endregion

# region forecast
# отвечает за сам процесс прогнозирования


final_df = transposition1  # получение преобразованной таблицы
grouped = final_df.groupby(['Регион продаж', 'Продукция'])  # выделение групп 'Регион продаж' + 'Продукция' из всей таблицы
outputlist = pd.DataFrame(columns = ['Регион продаж', 'Продукция', 'Дата Продажи', 'Кол-во продаж'])  # заранее создается dataframe для записи спрогнозированных в цикле значений
daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')  # диапазон дат. Определяется динамически и работает на малой выборке из 10 элементов и не должен ломаться, но не проверял на 3777 элементах. Поэтому если возникнут ошибки, строка ниже со статичными датами которые вручную меняются в коде точно не ломалась при обработке 3777 элементах
# daterng = pd.date_range(start='2023-06-01', end='2024-12-01', freq='MS')  # Здесь они вписаны вручную и проверены на 3777

print(len(grouped))  # вывод количества получившихся групп


isStationary = bool                      # используется только в блоке диффиренциации
isStationaryAfterDifferentiating = bool  # используется только в блоке диффиренциации
periodsAmount = 0                        # используется только в блоке диффиренциации


groupCounter = -1  # -1 т.к первая строка в цикле сразу прибавляет 1, и первая итерация будет записывать первое спрогнозированное значение в outputlist корректно. Индекс первого элемента в массиве = 0, а не 1. Сама переменная имеет два предназначения: служит индексом по которому записывается спрогнозированное значение в выходной массив, а также позволяет отслеживать на какой группе в данный момент производится прогноз

for (region, product), group in grouped:
    groupCounter += 1


    periodsAmount = 0     # используется только в блоке диффиренциации. Каждый цикл степень диффиренциации сбрасывается до 0
    isStationary = False  # используется только в блоке диффиренциации. Заранее предполагает что ряд нестационарен, затем проводится тест ряда на стационарность. И если ряд сразу является стационарным, он не будет обрабатываться, а сразу перейдет в блок ARIMA
    isStationaryAfterDifferentiating = False  # используется только в блоке диффиренциации


    forecast_dataframe = group[['Дата Продажи','Кол-во продаж']].copy()
    forecast_dataframe['Дата Продажи'] = pd.to_datetime(forecast_dataframe['Дата Продажи'])
    forecast_dataframe.set_index('Дата Продажи', inplace = True)
    originalTS =pd.Series(forecast_dataframe['Кол-во продаж'].tolist(), index = daterng)  # преобразование в тип pd.Series является необходимым условием для работы модели ARIMA
    diffirentiatedTS = pd.Series
    diffirentiatedTS = originalTS  # используется всегда. В случае если блок диффиренциации активен, diffirentiatedTS будет перезаписан при необходимости


    # region Differencing
    # блок диффиренцирования

    if (originalTS.eq(0).all() == True):  # если весь ряд значений состоит из нулей
        diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
        continue

    if ((adfuller(originalTS)[0] < adfuller(originalTS)[4]['1%']) == True): # raises warning if not enough non-zeroes to calculate.  // adfuller(originalTS) возвращает массив значений. Здесь для проверки использовалось сравнение реального и критического значения. adfuller(originalTS)[0] возвращает реальное значение; adfuller(originalTS)[4]['1%']) возвращает критическое значение, ['1%'] необходим для обработки получаемого числа чтобы избежать ошибки
        isStationary = True
        diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    else:
        isStationary = False
       
    if (isStationary == False):
        while (isStationaryAfterDifferentiating == False):
            if (periodsAmount == 13):  # в скобках находится крайнее значение количества элементов, при достижении которого диффиренциация больше не будет проводится. Например, при группе с кол-вом значений в 20, я всегда оставлял не меньше 7 (20-13=7) элементов, на основе которых ARIMA прогнозировала значения
                diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
                break
            else:  # пока не достигнуто крайнее значение количества элементов/ряд не стал стационарным (второй if в ветвлении try), процесс диффиренциации продолжается
                periodsAmount += 1
                try:
                    differenced = originalTS.diff(periods=periodsAmount).dropna()
                    # check if the differenced series is constant
                    if differenced.nunique() <= 1:
                        # skip this iteration if differenced series is constant
                        continue
                  
                    adf_result = adfuller(differenced)
                    if adf_result[0] < adf_result[4]['1%']:
                        isStationaryAfterDifferentiating = True
                        diffirentiatedTS = differenced
                        break
                except ValueError as e:
                    # log error and continue with next period
                    print(f"Error with period {periodsAmount} for {region}-{product}: {e}")
                    continue
    # endregion


    maxP = len(diffirentiatedTS-1)  # максимальный порядок авторегресии (p)
    maxQ = round(len(diffirentiatedTS) / 10)  # максимальное значение скользящего среднего (q); само максимальное значение находится по формуле описанной в строке

    print() # служит как "/n", позволяет легче замечать текущую группу которая обрабатыватеся в консоли вывода
    print()
    print()
    print(groupCounter)  # показывает текущую группу которая обрабатыватеся
    print()
    print()
    print()

    for p in range(0,maxP):
        for q in range(0, maxQ):
            if (q != 0):  # если это не первая итерация цикла подбора q то выполнение переходит в блок if
                fittedModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q))
                fittedModelQ1.initialize_approximate_diffuse()  # .initialize_approximate_diffuse() поправляет возникающую ошибку
                fittedModelQ = fittedModelQ1.fit()  # .fit() производит обучение модели на текущей группе значений
                if (fittedModelQ.aic < bestModelQ.aic):  # определяет какая модель лучше в сравнении по тесту Акаике; если она лучше предыдущей, то bestModelQ перезаписывается
                    bestModelQ = fittedModelQ
            else:
                if (maxQ >= 2):  # недописанное ветвление когда (Q >= 2), если в ряду/группе будет больше 20 значений
                    model3 = sm.tsa.arima.ARIMA
                    bestModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q))
                    bestModelQ1.initialize_approximate_diffuse()
                    bestModelQ = bestModelQ1.fit()
                else:
                    bestModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,q))
                    bestModelQ1.initialize_approximate_diffuse()
                    bestModelQ = bestModelQ1.fit()
        if (p != 0):  # если это не первая итерация цикла подбора p то выполнение переходит в блок if
          fittedModelP1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = 'Кол-во продаж')['Кол-во продаж'], order = (p,0,bestModelQ.model_orders['ma']))  # bestModelQ.model_orders['ma'] получает значение q
          fittedModelP1.initialize_approximate_diffuse()
          fittedModelP = fittedModelP1.fit()
          if (fittedModelP.aic < bestModelP.aic):  # такой же принцип как и в цикле подбора q
                bestModelP = fittedModelP
        else:  # если это первая итерация, то p = 0, соответственно можно использовать bestModelQ
          bestModelP = bestModelQ

    forecastValue = bestModelP.forecast(steps=1).iloc[0]
    forecastDate = bestModelP.forecast(steps=1).index[0]
    outputlist.loc[groupCounter] = [str(region), str(product), forecastDate, forecastValue]  # заполнение строки в выходной таблице
outputlist.to_excel("output.xlsx")
# endregion
