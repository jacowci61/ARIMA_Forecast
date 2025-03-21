# -*- coding: cp1251 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import forecast
# from statsmodels.tsa.arima.model import ARIMA

# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx')
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src_edited.xlsx')
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
print(inputTable)

inputTable = inputTable.melt(id_vars=['������ ������', '���������'], var_name='���� �������', value_name='���-�� ������')
inputTable['���� �������'] = pd.to_datetime(inputTable['���� �������'], format='%d.%m.%Y')

unique_regions = inputTable['������ ������'].unique()
unique_products = inputTable['���������'].unique()
unique_dates = pd.date_range(start=inputTable['���� �������'].min(), end=inputTable['���� �������'].max(), freq = 'MS')

cartesian_inputTable = pd.MultiIndex.from_product([unique_regions, unique_products, unique_dates] ,names = ['������ ������', '���������', '���� �������']).to_frame(index = False)
final_df = cartesian_inputTable.merge(inputTable, on = ['������ ������', '���������', '���� �������'], how = 'left')
print("\n")
print(pd.DataFrame(final_df))

print("\n")
print(pd.DataFrame(inputTable).head(200))# .sort_values(by=['������ ������', '���������', '���� �������']))

grouped = inputTable.groupby(['������ ������', '���������'])
print("\n")
print(pd.DataFrame(grouped))

outputlist = []
groupCounter = 0

# for (region, product), group in grouped:
#     forecast_dataframe = group[['���� �������','���-�� ������']].copy()
#     forecast_dataframe['���� �������'] = pd.to_datetime(forecast_dataframe['���� �������'])
#     forecast_dataframe.set_index('���� �������', inplace = True)
#     model = sm.tsa.arima.ARIMA(forecast_dataframe['���-�� ������'])
#     outputlist.append(model.fit().forecast(steps = 1))
#     groupCounter += 1

# outputTable = pd.DataFrame(outputlist)
# print(outputTable)
# print(groupCounter)