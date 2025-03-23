# -*- coding: cp1251 -*-
from collections import Counter
from itertools import combinations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import forecast
# from statsmodels.tsa.arima.model import ARIMA

inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src_edited.xlsx')
#inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
(inputTable[['Регион продаж','Продукция']].drop_duplicates()).to_excel("combinations_test.xlsx")
dates = inputTable.columns[2:]
print(dates[0])
print(dates[-2])
# print(inputTable.iloc[1].values(len(inputTable.columns)-1))

# inputTable.to_excel("read_test.xlsx")
# (inputTable.sort_values(by = 'Регион продаж')).to_excel("read_test.xlsx") sorting not needed

# inputTable['Регион продаж'] = pd.Categorical(inputTable['Регион продаж'], categories = inputTable['Регион продаж'].unique(), ordered = True)
# inputTable['Продукция'] = pd.Categorical(inputTable['Продукция'], categories = inputTable['Продукция'].unique(), ordered = True)

#inputTable1 = pd.melt(inputTable, id_vars=['Регион продаж', 'Продукция'], var_name='Дата продажи', value_name='Кол-во продаж')
#inputTable1.sort_values(['Регион продаж', 'Продукция'], key=lambda x: x.cat.codes)
#inputTable['Дата продажи'] = pd.to_datetime(inputTable['Дата продажи'], format='%d.%m.%Y')
#print(inputTable1)
#print(len(inputTable1))
#inputTable1.to_excel("transpose_test.xlsx")


# unique_regions = inputTable['Регион продаж'].unique()
# unique_products = inputTable['Продукция'].unique()
# print(len(unique_products))
# print(len(unique_regions))
# unique_dates = pd.date_range(start=inputTable['Дата продажи'].min(), end=inputTable['Дата продажи'].max(), freq = 'MS')

# cartesian_inputTable = pd.MultiIndex.from_product([unique_regions, unique_products, unique_dates] ,names = ['Регион продаж', 'Продукция', 'Дата продажи']).to_frame(index = False)
# final_df = cartesian_inputTable.merge(inputTable, on = ['Регион продаж', 'Продукция', 'Дата продажи'], how = 'left')
# print(len(cartesian_inputTable))
# print("\n")
# print(pd.DataFrame(final_df))

# print("\n")
# print(pd.DataFrame(inputTable))

# grouped = final_df.groupby(['Регион продаж', 'Продукция'])
# print("\n")
# print(pd.DataFrame(grouped))

# outputlist = pd.DataFrame(columns = ['Регион продаж', 'Продукция', 'Дата продажи', 'Кол-во продаж'])
# groupCounter = -1

# for (region, product), group in grouped:
#     groupCounter += 1 #5287 groups in summary
#     forecast_dataframe = group[['Дата продажи','Кол-во продаж']].copy()
#     forecast_dataframe['Дата продажи'] = pd.to_datetime(forecast_dataframe['Дата продажи'])
#     forecast_dataframe.set_index('Дата продажи', inplace = True)
#     model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'])
#     # outputlist.append(str(model.fit().forecast(steps = 1)).split(" "))
#     outputlist.loc[groupCounter] = [str(region), str(product), str(model.fit().forecast(steps = 1)).split(" ")[0], str(model.fit().forecast(steps = 1)).split(" ")[4].replace('\nFreq:', '')]

# outputlist.to_excel("output.xlsx")
# print(groupCounter)


# fix this ---------------
    # issue is in getting unique forecasted values to see if there are duplicates, because now amount of forecasted values and
    # amount of real values doesnt match
# outputtable = pd.DataFrame(outputlist)
# fix this ---------------

# print(outputtable)
# print(groupCounter)