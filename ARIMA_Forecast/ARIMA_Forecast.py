# -*- coding: cp1251 -*-
from collections import Counter
from datetime import datetime
from itertools import combinations
import string
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import forecast
# from statsmodels.tsa.arima.model import ARIMA


def convert_to_datetime(df):
    for col in df.columns[2:]:  # Skip first two columns
        try:
            # If column is string representation of date in column name
            if isinstance(col, str):
                df[col] = pd.to_datetime(col)
            
            # If column contains numeric values
            elif df[col].dtype in ['int64', 'float64']:
                df[col] = pd.to_datetime(df[col], unit='D')  # Assumes days since epoch
        except Exception as e:
            print(f"Conversion error for column {col}: {e}")
    return df

# region readingTable
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')

# delete headers and transpose salesValues, then merge row by row in ONE LONG COLUMN
salesValues = inputTable.drop(inputTable.columns[[0,1,-1]], axis = 1)
salesValuesLIST =[]
# print(len(salesValues.index))
for i in range(3777):
    salesValuesLIST.append(salesValues.iloc[i].tolist())
with open("salesvalues.txt", "w") as output:
    output.write(str(salesValuesLIST))

dates = inputTable.columns[2:]
date_start = pd.to_datetime(dates[0], dayfirst=True)
date_end = pd.to_datetime(dates[-1], dayfirst=True)

# Generate a continuous date range
daterng = pd.date_range(start="2023-06-01", end="2025-01-01", freq='MS')

# Convert date range to int64 timestamps
date_range = pd.DataFrame({'Дата продаж': daterng})

# Extract unique region-product pairs
unique_pairs = inputTable[['Регион продаж', 'Продукция']].drop_duplicates()


# Create all combinations using cross join
# WORKS!-----------------------------------------------------------------------------------------------------------------------------------
transposition1 = unique_pairs.merge(date_range, how="cross")
transposition1.to_excel("combinations_test.xlsx")
# WORKS!------------------------------------------------------------
# endregion

# region forecast
# final_df = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src2.xlsx')
# grouped = final_df.groupby(['Регион продаж', 'Продукция'])
# print("\n")
# print(pd.DataFrame(grouped))
# outputlist = pd.DataFrame(columns = ['Регион продаж', 'Продукция', 'Дата продажи', 'Кол-во продаж'])
# groupCounter = -1


# for (region, product), group in grouped:
#     groupCounter += 1
#     forecast_dataframe = group[['Дата Продажи','Кол-во продаж']].copy()
#     forecast_dataframe['Дата Продажи'] = pd.to_datetime(forecast_dataframe['Дата Продажи'])
#     forecast_dataframe.set_index('Дата Продажи', inplace = True)
#     model = sm.tsa.arima.ARIMA(forecast_dataframe['Кол-во продаж'])
#     # outputlist.append(str(model.fit().forecast(steps = 1)).split(" "))
#     print(str(region), str(product), str(model.fit().forecast(steps = 1)).split(" ")[0], str(model.fit().forecast(steps = 1)).split(" ")[4].replace('\nFreq:', ''))
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
# endregion