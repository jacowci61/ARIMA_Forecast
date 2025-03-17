# -*- coding: cp1251 -*-
import pandas as pd
# import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)

inputTable = inputTable.melt(id_vars=['Регион продаж', 'Продукция'], var_name='Дата продажи', value_name='Кол-во продаж')
inputTable['Дата продажи'] = pd.to_datetime(inputTable['Дата продажи'], format='%d.%m.%Y')
print(inputTable.head(50))

# print(inputTable.sort_values(by=['Регион продаж', 'Продукция','Дата продажи']).reset_index(drop = False).head(50))

grouped = inputTable.groupby(['Регион продаж', 'Продукция'])
for (region, product), group in grouped:
    print(region + " " + product)
    print(group)