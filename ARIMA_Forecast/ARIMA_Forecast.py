# -*- coding: cp1251 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import forecast
# from statsmodels.tsa.arima.model import ARIMA

# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src_edited.xlsx')
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
print(inputTable)

inputTable = inputTable.melt(id_vars=['Регион продаж', 'Продукция'], var_name='Дата продажи', value_name='Кол-во продаж')
inputTable['Дата продажи'] = pd.to_datetime(inputTable['Дата продажи'], format='%d.%m.%Y')

unique_regions = inputTable['Регион продаж'].unique()
unique_products = inputTable['Продукция'].unique()
unique_dates = pd.date_range(start=inputTable['Дата продажи'].min(), end=inputTable['Дата продажи'].max(), freq = 'MS')

cartesian_inputTable = pd.MultiIndex.from_product([unique_regions, unique_products, unique_dates] ,names = ['Регион продаж', 'Продукция', 'Дата продажи']).to_frame(index = False)
final_df = cartesian_inputTable.merge(inputTable, on = ['Регион продаж', 'Продукция', 'Дата продажи'], how = 'left')
print("\n")
print(pd.DataFrame(final_df))

print("\n")
print(pd.DataFrame(inputTable).head(200))# .sort_values(by=['Регион продаж', 'Продукция', 'Дата продажи']))

grouped = inputTable.groupby(['Регион продаж', 'Продукция'])
print("\n")
print(pd.DataFrame(grouped))

outputlist = []
groupCounter = 0

# for (region, product), group in grouped:
#     forecast_dataframe = group[['Дата продажи','Кол-во продаж']].copy()
#     forecast_dataframe['Дата продажи'] = pd.to_datetime(forecast_dataframe['Дата продажи'])
#     forecast_dataframe.set_index('Дата продажи', inplace = True)
#     model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'])
#     outputlist.append(model.fit().forecast(steps = 1))
#     groupCounter += 1

# outputTable = pd.DataFrame(outputlist)
# print(outputTable)
# print(groupCounter)