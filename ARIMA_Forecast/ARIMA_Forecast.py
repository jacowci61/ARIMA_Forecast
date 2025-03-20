# -*- coding: cp1251 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import forecast
# from statsmodels.tsa.arima.model import ARIMA

# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src_edited.xlsx')
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)

inputTable = inputTable.melt(id_vars=['Регион продаж', 'Продукция'], var_name='Дата продажи', value_name='Кол-во продаж')
inputTable['Дата продажи'] = pd.to_datetime(inputTable['Дата продажи'], format='%d.%m.%Y')

grouped = inputTable.groupby(['Регион продаж', 'Продукция'])
outputlist = []
groupCounter = 0

for (region, product), group in grouped:
    forecast_dataframe = group[['Дата продажи','Кол-во продаж']].copy()
    forecast_dataframe['Дата продажи'] = pd.to_datetime(forecast_dataframe['Дата продажи'])
    forecast_dataframe.set_index('Дата продажи', inplace = True)
    model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'])
    outputlist.append(model.fit().forecast(steps = 1))
    groupCounter += 1

outputTable = pd.DataFrame(outputlist)
print(outputTable)
print(groupCounter)