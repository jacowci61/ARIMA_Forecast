# -*- coding: cp1251 -*-
from collections import Counter
from datetime import datetime
from itertools import combinations
from _pytest.pathlib import bestrelpath
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.vector_ar.vecm import forecast
from pmdarima import auto_arima
from matplotlib import pyplot
import string
import pandas as pd
import numpy as np
import statsmodels.api as sm


def convert_to_datetime(df):
    for col in df.columns[2:]:  # ���������� ������ ��� ������� (������, �������� ���������), �.� ��� �� �������� �����
        try:
            if isinstance(col, str):  # ���� ��� ������ � ������� � ����� �������� string
                df[col] = pd.to_datetime(col)
            
            elif df[col].dtype in ['int64', 'float64']:  # ���� ��� ������ � ������� � ����� �������� int/float/����� ����� ������ ������� �������� ������
                df[col] = pd.to_datetime(df[col], unit='D')  # assumes days since epoch
        except Exception as e:
            print(f"Conversion error for column {col}: {e}")
    return df

# region readingTable
# �������� �� ��������� ������ � �������


# ���������� ����� ����� ������� ����� ������:
# ���� ���� �� ��������� � ����� ����� � ������ ��������� (.py), ����� ������������ ���������� ����: inputTable = pd.read_excel('D:...\�������������\�������������.xlsx')
# ���� ���� ��������� � ����� ����� � ������ ��������� (.py): inputTable = pd.read_excel('D:...\�������������\�������������.xlsx')
inputTable = pd.read_excel('D:\PocoX3\Work\Involux\���������������\src.xlsx')


salesValues = inputTable.drop(inputTable.columns[[0,1,-1,-2]], axis = 1)
# �������� ������ ������ .drop ��������� �� ���� ������� (inputTable) � ������� �������� (inputTable.columns[[]]) ������� ����� �������.
# ����� ������ .columns[[]] ����������� ������� �������� ������� ����� �������. [[...,-1,-2]] ������� ������� � ����� �������
salesValuesLIST =[]

for i in range(salesValues.shape[0]):  # ��������� �������� ������
    salesValuesLIST.extend(salesValues.iloc[i].tolist())
salesValuesDF = pd.DataFrame({'���-�� ������' : salesValuesLIST})

dates = inputTable.columns[2:]  # ���������� ��������� ��� �� ���� ��������, ����� ������ ����
date_start = pd.to_datetime(dates[0], dayfirst=True)
date_end = pd.to_datetime(dates[-1], dayfirst=True)  # dates[-1] �������� ��������� ������� � ������� ���� ���

daterng = pd.date_range(start=salesValues.columns[0], end=salesValues.columns[-1], freq='MS')

date_range = pd.DataFrame({'���� �������': daterng})

unique_pairs = inputTable[['������ ������', '���������']].drop_duplicates()  # ��������� ���������� ���������� '������ ������' + '���������'

# ��������� ��� ������ �������� �� ���������������� ��� ���������� ������� �� "wide" � "long" ������, ������� ��������� ��� ������� ��������� �����
transposition1 = unique_pairs.merge(date_range, how="cross")
transposition1 = pd.concat([transposition1, salesValuesDF], axis = 1)
transposition1.to_excel("combinations_test.xlsx")
# endregion

# region forecast
# �������� �� ��� ������� ���������������


final_df = transposition1  # ��������� ��������������� �������
grouped = final_df.groupby(['������ ������', '���������'])  # ��������� ����� '������ ������' + '���������' �� ���� �������
outputlist = pd.DataFrame(columns = ['������ ������', '���������', '���� �������', '���-�� ������'])  # ������� ��������� dataframe ��� ������ ����������������� � ����� ��������
daterng = pd.date_range(start='2023-06-01', end='2024-12-01', freq='MS')  # �������� ���. ����� ��� ������� �������, �� ����� ������� ������������ ��������� ������� � ���������� inputTable.columns[2] � inputTable.columns[-1]

print(len(grouped))  # ����� ���������� ������������ �����


# isStationary = bool                      # ������������ ������ � ����� ��������������
# isStationaryAfterDifferentiating = bool  # ������������ ������ � ����� ��������������
# periodsAmount = 0                        # ������������ ������ � ����� ��������������


groupCounter = -1  # -1 �.� ������ ������ � ����� ����� ���������� 1, � ������ �������� ����� ���������� ������ ����������������� �������� � outputlist ���������. ������ ������� �������� � ������� = 0, � �� 1. ���� ���������� ����� ��� ��������������: ������ �������� �� �������� ������������ ����������������� �������� � �������� ������, � ����� ��������� ����������� �� ����� ������ � ������ ������ ������������ �������

for (region, product), group in grouped:
    groupCounter += 1


    # periodsAmount = 0     # ������������ ������ � ����� ��������������. ������ ���� ������� �������������� ������������ �� 0
    # isStationary = False  # ������������ ������ � ����� ��������������. ������� ������������ ��� ��� �������������, ����� ���������� ���� ���� �� ��������������. � ���� ��� ����� �������� ������������, �� �� ����� ��������������, � ����� �������� � ���� ARIMA
    # isStationaryAfterDifferentiating = False  # ������������ ������ � ����� ��������������


    forecast_dataframe = group[['���� �������','���-�� ������']].copy()
    forecast_dataframe['���� �������'] = pd.to_datetime(forecast_dataframe['���� �������'])
    forecast_dataframe.set_index('���� �������', inplace = True)
    originalTS =pd.Series(forecast_dataframe['���-�� ������'].tolist(), index = daterng)  # �������������� � ��� pd.Series �������� ����������� �������� ��� ������ ������ ARIMA
    diffirentiatedTS = pd.Series
    diffirentiatedTS = originalTS  # ������������ ������. � ������ ���� ���� �������������� �������, diffirentiatedTS ����� ����������� ��� �������������


    # region Differencing
    # ���� �����������������

    # if (originalTS.eq(0).all() == True):  # ���� ���� ��� �������� ������� �� �����
    #     diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    #     continue

    # if ((adfuller(originalTS)[0] < adfuller(originalTS)[4]['1%']) == True): # raises warning if not enough non-zeroes to calculate.  // adfuller(originalTS) ���������� ������ ��������. ����� ��� �������� �������������� ��������� ��������� � ������������ ��������. adfuller(originalTS)[0] ���������� �������� ��������; adfuller(originalTS)[4]['1%']) ���������� ����������� ��������, ['1%'] ��������� ��� ��������� ����������� ����� ����� �������� ������
    #     isStationary = True
    #     diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    # else:
    #     isStationary = False
        
    # if (isStationary == False):
    #     while (isStationaryAfterDifferentiating == False):
    #         if (periodsAmount == 13):  # � ������� ��������� ������� �������� ���������� ���������, ��� ���������� �������� �������������� ������ �� ����� ����������. ��������, ��� ������ � ���-��� �������� � 20, � ������ �������� �� ������ 7 (20-13=7) ���������, �� ������ ������� ARIMA �������������� ��������
    #             diffirentiatedTS = originalTS.diff(periodsAmount).dropna()
    #             break
    #         else:  # ���� �� ���������� ������� �������� ���������� ���������/��� �� ���� ������������ (������ if � ��������� try), ������� �������������� ������������
    #             periodsAmount += 1
    #             try:
    #                 differenced = originalTS.diff(periods=periodsAmount).dropna()
    #                 # check if the differenced series is constant
    #                 if differenced.nunique() <= 1:
    #                     # skip this iteration if differenced series is constant
    #                     continue
                    
    #                 adf_result = adfuller(differenced)
    #                 if adf_result[0] < adf_result[4]['1%']:
    #                     isStationaryAfterDifferentiating = True
    #                     diffirentiatedTS = differenced
    #                     break
    #             except ValueError as e:
    #                 # log error and continue with next period
    #                 print(f"Error with period {periodsAmount} for {region}-{product}: {e}")
    #                 continue
    # endregion


    maxP = len(diffirentiatedTS-1)  # ������������ ������� ������������ (p)
    maxQ = round(len(diffirentiatedTS) / 10)  # ������������ �������� ����������� �������� (q); ���� ������������ �������� ��������� �� ������� ��������� � ������

    print() # ������ ��� "/n", ��������� ����� �������� ������� ������ ������� ��������������
    print()
    print()

    print(groupCounter)  # ���������� ������� ������ ������� ��������������

    print()
    print()
    print()

    for p in range(0,maxP):
        for q in range(0, maxQ):
            if (q != 0):  # ���� ��� �� ������ �������� ����� ������� q �� ���������� ��������� � ���� if
                fittedModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = '���-�� ������')['���-�� ������'], order = (p,0,q))
                fittedModelQ1.initialize_approximate_diffuse()  # .initialize_approximate_diffuse() ���������� ����������� ������
                fittedModelQ = fittedModelQ1.fit()  # .fit() ���������� �������� ������ �� ������� ������ ��������
                if (fittedModelQ.aic < bestModelQ.aic):  # ���������� ����� ������ ����� � ��������� �� ����� ������; ���� ��� ����� ����������, �� bestModelQ ����������������
                    bestModelQ = fittedModelQ
            else:
                if (maxQ >= 2):  # ���� � ����/������ ����� ������ 20 ��������, ����� Q ����� ������ ������
                    model3 = sm.tsa.arima.ARIMA  # buffer for best model if Q >=2. Not the case for current range of values
                    bestModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = '���-�� ������')['���-�� ������'], order = (p,0,q))
                    bestModelQ1.initialize_approximate_diffuse()
                    bestModelQ = bestModelQ1.fit()
                else:
                    bestModelQ1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = '���-�� ������')['���-�� ������'], order = (p,0,q))
                    bestModelQ1.initialize_approximate_diffuse()
                    bestModelQ = bestModelQ1.fit()
        if (p != 0):  # ���� ��� �� ������ �������� ����� ������� p �� ���������� ��������� � ���� if
          fittedModelP1 = sm.tsa.arima.ARIMA(diffirentiatedTS.to_frame(name = '���-�� ������')['���-�� ������'], order = (p,0,bestModelQ.model_orders['ma']))  # bestModelQ.model_orders['ma'] �������� �������� q
          fittedModelP1.initialize_approximate_diffuse()
          fittedModelP = fittedModelP1.fit()
          if (fittedModelP.aic < bestModelP.aic):  # ����� �� ������� ��� � � ����� ������� q
                bestModelP = fittedModelP
        else:
          bestModelP = bestModelQ

    forecastValue = bestModelP.forecast(steps=1).iloc[0]
    forecastDate = bestModelP.forecast(steps=1).index[0]
    outputlist.loc[groupCounter] = [str(region), str(product), forecastDate, forecastValue]  # ���������� ������ � �������� �������
outputlist.to_excel("output.xlsx")
# endregion