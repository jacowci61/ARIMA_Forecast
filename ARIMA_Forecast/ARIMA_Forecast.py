# -*- coding: cp1251 -*-
import pandas as pd
import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA

# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx')
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src_edited.xlsx')
inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)

inputTable = inputTable.melt(id_vars=['������ ������', '���������'], var_name='���� �������', value_name='���-�� ������')
inputTable['���� �������'] = pd.to_datetime(inputTable['���� �������'], format='%d.%m.%Y')
# print(inputTable.head(50))

# print(inputTable.sort_values(by=['������ ������', '���������','���� �������']).reset_index(drop = False).head(50))

grouped = inputTable.groupby(['������ ������', '���������'])

for (region, product), group in grouped:
    forecast_dataframe = group[['���� �������','���-�� ������']].copy()
    print(forecast_dataframe.index.dtype)
    print(forecast_dataframe['���� �������'].dtype)
    # forecast_dataframe.index = forecast_dataframe.DatetimeIndex(forecast_dataframe.index).to_period('M')
    # print(forecast_dataframe.head(50))
    model = sm.tsa.arima.ARIMA(forecast_dataframe['���-�� ������'], order = (1,1,1))
    model_fit = model.fit()
    print(model_fit.summary())

# for (region, product), group in grouped:
#      print(region + " " + product)
#      print(group)