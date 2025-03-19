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
# print(inputTable.head(50))

# print(inputTable.sort_values(by=['Регион продаж', 'Продукция','Дата продажи']).reset_index(drop = False).head(50))

grouped = inputTable.groupby(['Регион продаж', 'Продукция'])

for (region, product), group in grouped:
    forecast_dataframe = group[['Дата продажи','Кол-во продаж']].copy()
    # print(forecast_dataframe)
    forecast_dataframe['Дата продажи'] = pd.to_datetime(forecast_dataframe['Дата продажи'])
    forecast_dataframe.set_index('Дата продажи', inplace = True)
    print(forecast_dataframe)
    #forecast_data = np.array(forecast_dataframe)
    # print(forecast_data)
    model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'], order = (1,1,1))
    #model_fit = model.fit()
    print(model.fit().forecast(1))
    # forecast_dataframe['Дата продажи'] = pd.to_datetime(inputTable['Дата продажи'], format='%d.%m.%Y')
    # forecast_dataframe = forecast_dataframe.to_numpy()
    # print(forecast_dataframe.index.dtype)
    # print(forecast_dataframe['Дата продажи'].dtype)
    # forecast_dataframe.index = forecast_dataframe.DatetimeIndex(forecast_dataframe.index).to_period('M')
    # print(forecast_dataframe.head(50))
    # print(forecast_dataframe)
    # model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'], order = (1,1,1))
    # model_fit = model.fit()

# for (region, product), group in grouped:
#      print(region + " " + product)
#      print(group) 