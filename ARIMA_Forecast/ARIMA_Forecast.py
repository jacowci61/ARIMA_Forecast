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
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx')
print("111111111111111111111")
print(inputTable.dtypes)
print(inputTable.info())
# inputTable.columns = [pd.to_datetime(col, dayfirst=True, errors='ignore') if isinstance(col, str) else col for col in inputTable.columns]
print("2222222222222222222")
inputTable = convert_to_datetime(inputTable)
print(inputTable.dtypes)
print(inputTable.info())
# read date as object | inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx', parse_dates = inputTable.columns[2:].tolist())#, date_format ='%d.%m.%Y')
# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src_edited.xlsx')
#inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
# inputTable.drop(inputTable.columns[len(inputTable.columns)-1], axis = 1, inplace = True)
# (inputTable[['������ ������','���������']].drop_duplicates()).to_excel("combinations_test.xlsx")
# inputTable.columns = list(inputTable.columns[:2]) + list(pd.to_datetime(inputTable.columns[2:]))

# ------new attempt------
# dates = inputTable.columns[2:]
# date_start = pd.to_datetime(dates[0], dayfirst=True)
# date_end = pd.to_datetime(dates[-1], dayfirst=True)

# # Generate a continuous date range
# daterng = pd.date_range(start=date_start, end=date_end, freq='MS')

# # Convert date range to int64 timestamps
# date_range = pd.DataFrame({'���� ������': daterng.astype('int64')})

# # Extract unique region-product pairs
# unique_pairs = inputTable[['������ ������', '���������']].drop_duplicates()

# # Convert '���� ������' column in inputTable to int64 for merging
# # inputTable['���� ������'] = pd.to_datetime(inputTable['���� ������'], dayfirst=True).astype('int64')

# # Create all combinations using cross join
# transposition1 = unique_pairs.merge(date_range, how="cross")

# # Merge back to original table using int64 timestamps
# transposition2 = transposition1.merge(inputTable, on=['������ ������', '���������', '���� ������'], how="left")

# print(transposition2.head())
# ------new attempt------

new_columns = []
for col in inputTable.columns:
    try:
        new_columns.append(pd.to_datetime(col))
    except:
        new_columns.append(col)  # Keep non-date columns unchanged

# Assign updated column names
inputTable.columns = new_columns
for col in inputTable.select_dtypes(include=['object']).columns:
    try:
        inputTable[col] = pd.to_datetime(inputTable[col])
    except:
        pass  # Skip columns that cannot be converted
print(inputTable)
print("333333333333333333333333333")

# Display results


# inputTable.columns = [datetime(i) if isinstance(i, np.int64) else i for i in inputTable.columns]
# inputTable.iloc[2:] = pd.to_datetime(inputTable.columns[2:].to_list())
# inputTable.iloc[2:].replace(pd.to_datetime(inputTable.columns[2:].to_list()))
inputTable['2025-01-01 00:00:00'] = pd.to_datetime(inputTable['2025-01-01 00:00:00'])
print(inputTable.dtypes)
# print(inputTable.columns)

# new_df = pd.DataFrame()
# new_df['������ ������'] = result_df['������ ������']
# new_df['���������'] = result_df['���������']
# for date_col in result_df.columns[2:]:
#     for idx, row in result_df.iterrows():
#         # Create a datetime column with the integer value
#         new_df.at[idx, date_col] = pd.to_datetime(date_col)
# print(inputTable.dtypes)

# dates = inputTable.columns[2:]
# daterng =  pd.date_range(start = pd.to_datetime(dates[0]), end = pd.to_datetime(dates[-2]), freq = 'MS')
# date_range = pd.DataFrame({ '���� ������' : pd.to_datetime(daterng)})
# print(daterng)
# print(date_range)
# unique_pairs = inputTable[['������ ������','���������']].drop_duplicates()
# transposition1 = unique_pairs.merge(pd.to_datetime(date_range.stack()).unstack(), how = "cross")
# transposition1['���� ������'] = pd.to_datetime(transposition1['���� ������'])
# print("\n and some other letters")
# print(transposition1.dtypes)
# print("\n and some other letters")
# print(inputTable.dtypes)
# transposition2 = transposition1.merge(inputTable, on = ['������ ������','���������', '���� ������'], how  = "left")
# transposition2.to_excel("combinations_test.xlsx")


# ---------------------------------------------------------------------------------------------------------------





# print(inputTable.iloc[1].values(len(inputTable.columns)-1))

# inputTable.to_excel("read_test.xlsx")
# (inputTable.sort_values(by = '������ ������')).to_excel("read_test.xlsx") sorting not needed

# inputTable['������ ������'] = pd.Categorical(inputTable['������ ������'], categories = inputTable['������ ������'].unique(), ordered = True)
# inputTable['���������'] = pd.Categorical(inputTable['���������'], categories = inputTable['���������'].unique(), ordered = True)

#inputTable1 = pd.melt(inputTable, id_vars=['������ ������', '���������'], var_name='���� �������', value_name='���-�� ������')
#inputTable1.sort_values(['������ ������', '���������'], key=lambda x: x.cat.codes)
#inputTable['���� �������'] = pd.to_datetime(inputTable['���� �������'], format='%d.%m.%Y')
#print(inputTable1)
#print(len(inputTable1))
#inputTable1.to_excel("transpose_test.xlsx")


# unique_regions = inputTable['������ ������'].unique()
# unique_products = inputTable['���������'].unique()
# print(len(unique_products))
# print(len(unique_regions))
# unique_dates = pd.date_range(start=inputTable['���� �������'].min(), end=inputTable['���� �������'].max(), freq = 'MS')

# cartesian_inputTable = pd.MultiIndex.from_product([unique_regions, unique_products, unique_dates] ,names = ['������ ������', '���������', '���� �������']).to_frame(index = False)
# final_df = cartesian_inputTable.merge(inputTable, on = ['������ ������', '���������', '���� �������'], how = 'left')
# print(len(cartesian_inputTable))
# print("\n")
# print(pd.DataFrame(final_df))

# print("\n")
# print(pd.DataFrame(inputTable))


# ---------------------------------------------------------------------------------------------------------------

# final_df = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src2.xlsx')
# grouped = final_df.groupby(['������ ������', '���������'])
# print("\n")
# print(pd.DataFrame(grouped))

# outputlist = pd.DataFrame(columns = ['������ ������', '���������', '���� �������', '���-�� ������'])
# groupCounter = -1

# for (region, product), group in grouped:
#     groupCounter += 1
#     forecast_dataframe = group[['���� �������','���-�� ������']].copy()
#     forecast_dataframe['���� �������'] = pd.to_datetime(forecast_dataframe['���� �������'])
#     forecast_dataframe.set_index('���� �������', inplace = True)
#     model = sm.tsa.arima.ARIMA(forecast_dataframe['���-�� ������'])
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