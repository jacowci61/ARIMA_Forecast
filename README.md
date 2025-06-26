# Theory behind the project
Uses ARIMA model, processes data accordingly to Box-Jenkins methodology, except differencing of time series. It's bugged, works separately fine, but united with forecasting using ARIMA model itself it fails. So for now it's commented out
# Input/Output explanations
## Input table looks like this:
![image](https://github.com/user-attachments/assets/fcc07f98-b8f9-4dd6-843a-f2a8c41ab4d2)
- this is "wide" format of table. Program transforms it to "long" format, so time-series analysis can be performed
## Output table looks like this:
![image](https://github.com/user-attachments/assets/c8807583-fc58-4e49-9870-5ca559051571)
- to obtain real value, excel function 'rounddown()' can be used to extract integer part of a number (e.g 1.7 -> 1)
- non-positive values can be replaced with zeroes
