# -*- coding: cp1251 -*-
import pandas as pd
# import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx')
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)

inputTable = inputTable.melt(id_vars=['������ ������', '���������'], var_name='���� �������', value_name='���-�� ������')
inputTable['���� �������'] = pd.to_datetime(inputTable['���� �������'], format='%d.%m.%Y')
print(inputTable.head(50))

# print(inputTable.sort_values(by=['������ ������', '���������','���� �������']).reset_index(drop = False).head(50))

grouped = inputTable.groupby(['������ ������', '���������'])
for (region, product), group in grouped:
    print(region + " " + product)
    print(group)