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

# ---------------------------------------------------------------------------------------------------------------
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx')
# print("111111111111111111111")
# print(inputTable.dtypes)
# print(inputTable.info())
# inputTable.columns = [pd.to_datetime(col, dayfirst=True, errors='ignore') if isinstance(col, str) else col for col in inputTable.columns]
# print("2222222222222222222")
inputTable = convert_to_datetime(inputTable)
print(inputTable.dtypes)
print(inputTable.info())

# test transpositions

# regions = inputTable['Регион продаж'].unique()
# products = inputTable['Продукция'].unique()
# dates = pd.date_range(start="2023-01-01", end="2024-12-01", freq='MS')

# # Create a DataFrame with all combinations (Cartesian Product)
# cartesian_df = pd.MultiIndex.from_product([regions, products, dates], 
# names=['Регион продаж', 'Продукция', 'Дата продажи']).to_frame(index=False)

# # Merge with the original data to retain sales values, filling missing sales with 0 or NaN
# final_df = cartesian_df.merge(inputTable, on=['Регион продаж', 'Продукция', 'Дата продажи'], how='left')

# # Fill missing sales values (optional: use 0 or leave as NaN)
# final_df['Кол-во продаж'].fillna(0, inplace=True)

# # Save to Excel or print output
# # final_df.to_excel("output.xlsx", index=False)
# print(final_df.head(50))


# read date as object | inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src.xlsx', parse_dates = inputTable.columns[2:].tolist())#, date_format ='%d.%m.%Y')
# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\Прогнозирование\src_edited.xlsx')
#inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
# inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
# (inputTable[['Регион продаж','Продукция']].drop_duplicates()).to_excel("combinations_test.xlsx")
# inputTable.columns = list(inputTable.columns[:2]) + list(pd.to_datetime(inputTable.columns[2:]))

# ------new attempt1------
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
# WORKS!-----------------------------------------------------------------------------------------------------------------------------------


# Merge back to original table using int64 timestamps
# transposition2 = transposition1.merge(inputTable, on=['Регион продаж', 'Продукция', 'Дата продаж'], how="left")
# ------------


# ------new attempt2------
# new_df = pd.DataFrame()
# new_df['Регион продаж'] = inputTable['Регион продаж']
# new_df['Продукция'] = inputTable['Продукция']
# for date_col in inputTable.columns[2:]:
#     for idx, row in inputTable.iterrows():
#         # Create a datetime column with the integer value
#         new_df.at[idx, date_col] = pd.to_datetime(date_col)
# print(inputTable.dtypes)

# dates = inputTable.columns[2:]
# daterng =  pd.date_range(start = pd.to_datetime(dates[0]), end = pd.to_datetime(dates[-2]), freq = 'MS')
# date_range = pd.DataFrame({ 'Дата продаж' : pd.to_datetime(daterng)})
# print(daterng)
# print(date_range)
# unique_pairs = inputTable[['Регион продаж','Продукция']].drop_duplicates()
# transposition1 = unique_pairs.merge(pd.to_datetime(date_range.stack()).unstack(), how = "cross")
# transposition1['Дата продаж'] = pd.to_datetime(transposition1['Дата продаж'])
# print("\n and some other letters")
# print(transposition1.dtypes)
# print("\n and some other letters")
# print(inputTable.dtypes)
# transposition2 = transposition1.merge(inputTable, on = ['Регион продаж','Продукция', 'Дата продаж'], how  = "left")
# transposition2.to_excel("combinations_test.xlsx")
# ---------------------------------------------------------------------------------------------------------------





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


# ---------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------



# print(groupCounter)


# fix this ---------------
    # issue is in getting unique forecasted values to see if there are duplicates, because now amount of forecasted values and
    # amount of real values doesnt match
# outputtable = pd.DataFrame(outputlist)
# fix this ---------------

# print(outputtable)
# print(groupCounter)