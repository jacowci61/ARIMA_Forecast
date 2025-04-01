# -*- coding: cp1251 -*-
from collections import Counter
from datetime import datetime
from itertools import combinations
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.vector_ar.vecm import forecast
from pmdarima import auto_arima
from matplotlib import pyplot
import string
import pandas as pd
import numpy as np
import statsmodels.api as sm
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
# inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx')
# salesValues = inputTable.drop(inputTable.columns[[0,1,-1]], axis = 1)
# salesValuesLIST =[]

# for i in range(salesValues.shape[0]):
#     salesValuesLIST.extend(salesValues.iloc[i].tolist())
# salesValuesDF = pd.DataFrame({'���-�� ������' : salesValuesLIST})

# dates = inputTable.columns[2:]
# date_start = pd.to_datetime(dates[0], dayfirst=True)
# date_end = pd.to_datetime(dates[-1], dayfirst=True)

# daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')

# date_range = pd.DataFrame({'���� �������': daterng})

# # Extract unique region-product pairs
# unique_pairs = inputTable[['������ ������', '���������']].drop_duplicates()

# transposition1 = unique_pairs.merge(date_range, how="cross")
# transposition1 = pd.concat([transposition1, salesValuesDF], axis = 1)
# transposition1.to_excel("combinations_test.xlsx")
# endregion

# region forecast
final_df = pd.read_excel('src2.xlsx')
# final_df = transposition1
grouped = final_df.groupby(['������ ������', '���������'])
outputlist = pd.DataFrame(columns = ['������ ������', '���������', '���� �������', '���-�� ������'])
daterng = pd.date_range(start='2023-06-01', end='2025-01-01', freq='MS')

isStationary = bool
isStationaryAfterDifferentiating = bool
periodsAmount = 0
groupCounter = -1

for (region, product), group in grouped:
    forecast_dataframe = group[['���� �������','���-�� ������']].copy()
    forecast_dataframe['���� �������'] = pd.to_datetime(forecast_dataframe['���� �������'])
    forecast_dataframe.set_index('���� �������', inplace = True)
    originalTS =pd.Series(forecast_dataframe['���-�� ������'].tolist(), index = daterng)
    if (originalTS.eq(0).all() == True):
        print(f"HIT! ------------------------------------")
        continue
    groupCounter += 1
    periodsAmount = 0
    isStationary = False
    isStationaryAfterDifferentiating = False
    # print(adfuller(originalTS))
    if ((adfuller(originalTS)[0] < adfuller(originalTS)[4]['1%']) == True): # raises warning if not enough non-zeroes to calculate
        isStationary = True
    else:
        isStationary = False
        
    if (isStationary == False):
        while (isStationaryAfterDifferentiating == False):
            if (periodsAmount == 10):
                print(f"Exceeded limit!")
                break
            else:
                periodsAmount += 1
                if ((adfuller(originalTS.diff(periods = periodsAmount).dropna())[0] < adfuller(originalTS.diff(periods = periodsAmount).dropna())[4]['1%']) == True):
                    isStationaryAfterDifferentiating = True
                    print(f"Success on {periodsAmount} difference")
                    break
    # differencedTS = (originalTS).diff(periods = 1)
    # print(adfuller(differencedTS.dropna()))
    # print(adfuller(differencedTS.diff(periods = 2).dropna()))
    # print(adfuller(differencedTS.diff(periods = 3).dropna()))
    # print(adfuller(differencedTS.diff(periods = 4).dropna()))
    # print(adfuller(differencedTS.diff(periods = 5).dropna()))
    # print(adfuller(differencedTS.diff(periods = 6).dropna()))
    # print(adfuller(differencedTS.diff(periods = 7).dropna()))
    # print(adfuller(differencedTS.diff(periods = 8).dropna()))
    # print(adfuller(differencedTS.diff(periods = 9).dropna()))
    # print(adfuller(differencedTS.diff(periods = 10).dropna()))
    # print(adfuller(differencedTS.diff(periods = 11).dropna()))
    # print(adfuller(differencedTS.diff(periods = 12).dropna()))
    # print(differencedTS.diff(periods = 12).dropna())
    
    
    #originalTS.plot(color = "red")
    #differencedTS.dropna().plot(color = "blue")
    #pyplot.show()



    # region Forecasting
    #p,d,q = auto_arima(pd.Series(forecast_dataframe['���-�� ������'].tolist(), index = daterng), seasonal = False, trace = True, stepwise = True, max_p = 20, max_d = 20, max_q = 20, maxiter = 100).order
    # model = sm.tsa.arima.ARIMA(forecast_dataframe['���-�� ������'], order = (p,d,q))
    # outputlist.append(str(model.fit().forecast(steps = 1)).split(" "))
    # print(str(region), str(product), str(model.fit().forecast(steps = 1)).split(" ")[0], str(model.fit().forecast(steps = 1)).split(" ")[4].replace('\nFreq:', ''))
    #outputlist.loc[groupCounter] = [str(region), str(product), str(model.fit().forecast(steps = 1)).split(" ")[0], str(model.fit().forecast(steps = 1)).split(" ")[4].replace('\nFreq:', '')]
#outputlist.to_excel("output.xlsx")
# endregion