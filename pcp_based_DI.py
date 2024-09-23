# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:43:22 2024

@author: Deep_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\Deep_\Downloads\Kaggle\Drought_Indices\Data\prcphq.046037.month.txt", 
                   sep=r"\s+", skiprows=1, usecols=[1, 2], parse_dates=True, index_col=0, names=["Date", "Rain"])

data['Rain_6'] = data['Rain'].rolling(6).sum()  # 6-month cumulative rainfall
df_6mon = data[['Rain_6']].dropna()

def calculate_by_month(df, calc_function):
    df['Result'] = np.nan
    for imon in np.arange(1, 13):
        sinds = df.index.month == imon
        x = df[sinds]
        result = calc_function(x)
        
        # Check if the passed function is calc_rai
        if calc_function.__name__ in ('calc_rai', 'calc_edi', 'calc_spi'):
            df.loc[sinds, 'Result'] = result.values  # No slicing
        else:
            df.loc[sinds, 'Result'] = result.values[:, 0]  # With slicing
            
    # Replace NaN values in 'Result' with 0
    df['Result'].fillna(0, inplace=True)
            
    return df['Result']

# Percent of Normal Index (PNI)
def calc_pni(x):
    return (x / x['1981':'2010'].mean()) * 100.0


# Calculate Decile Index (DI)
def calc_di(x):
    return (x.rank() - 1.0) / (len(x) - 1.0) * 100.0

# Calculate Hutchinson Drought Severity Index (HDI)
def calc_hdi(x):
    y = (x.rank() - 1.0) / (len(x) - 1.0)
    return 8.0 * (y - 0.5)

# Calculate Z-Score Index (ZSI)
def calc_zsi(x):
    return (x - x.mean()) / x.std()

# Calculate China-Z Index (CZI)
def calc_czi(x):
    zsi = (x - x.mean()) / x.std()
    cs = np.power(zsi, 3) / len(x)
    return 6.0 / cs * np.power((cs / 2.0 * zsi + 1.0), 1.0 / 3.0) - 6.0 / cs + cs / 6.0

# Calculate Modified China-Z Index (MCZI)
def calc_mczi(x):
    zsi = (x - x.median()) / x.std()
    cs = np.power(zsi, 3) / len(x)
    return 6.0 / cs * np.power((cs / 2.0 * zsi + 1.0), 1.0 / 3.0) - 6.0 / cs + cs / 6.0



# Calculate Rainfall Anomaly Index (RAI)
def calc_rai(x):
    x1 = x.copy().sort_values(by='Rain_6', ascending=False)
    x_avg = x1['Rain_6'].mean()
    mx_avg = x1['Rain_6'].head(10).mean()
    mn_avg = x1['Rain_6'].tail(10).mean()
    anom = x['Rain_6'] - x_avg

    rai_plus = 3.0 * anom[anom >= 0] / (mx_avg - x_avg)
    rai_minus = -3.0 * anom[anom < 0] / (mn_avg - x_avg)
    y = x.copy()
    y.loc[anom >= 0, 'RAI'] = rai_plus.values
    y.loc[anom < 0, 'RAI'] = rai_minus.values
    return y['RAI']


def calc_edi(x):
    # Create a copy of the DataFrame slice
    x = x.copy()
    
    # Calculate cumulative precipitation
    x['Cumulative_Rain'] = x['Rain_6'].cumsum()
    
    # Calculate the average and standard deviation of cumulative precipitation
    mean_precip = x['Cumulative_Rain'].mean()
    std_precip = x['Cumulative_Rain'].std()
    
    # Calculate EDI
    edi = (x['Cumulative_Rain'] - mean_precip) / std_precip
    return edi


# Calculate Standardized Precipitation Index (SPI)
def calc_spi(x, scale=1):
    # Calculate the rolling sum of precipitation for the specified scale
    rolling_sum = x['Rain_6'].rolling(window=scale).sum()
    
    # Fit a normal distribution to the data
    mean = rolling_sum.mean()
    std = rolling_sum.std()
    
    # Calculate SPI
    spi = (rolling_sum - mean) / std
    return spi

df_6mon['PNI'] = calculate_by_month(df_6mon, calc_pni)
df_6mon['Prob'] = calculate_by_month(df_6mon, calc_di)
df_6mon['HDI'] = calculate_by_month(df_6mon, calc_hdi)
df_6mon['ZSI'] = calculate_by_month(df_6mon, calc_zsi)
df_6mon['CZI'] = calculate_by_month(df_6mon, calc_czi)
df_6mon['MCZI'] = calculate_by_month(df_6mon, calc_mczi)
df_6mon['RAI'] = calculate_by_month(df_6mon, calc_rai)
df_6mon['EDI'] = calculate_by_month(df_6mon, calc_edi)
df_6mon['SPI'] = calculate_by_month(df_6mon, calc_spi)



# Percent of Normal Index (PNI)
plt.figure(figsize=(15, 7))
df_6mon['PNI'].plot(color='b', linewidth=2)
plt.axhline(130, linestyle='--', color='g', label='Above Normal')
plt.axhline(80, linestyle='--', color='r', label='Below Normal')
plt.title('Six-Monthly Percent of Normal Index (PNI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylim(0, 500)
plt.ylabel('PNI (%)')
plt.legend()
plt.show()

# Decile Index (DI)
plt.figure(figsize=(15, 7))
df_6mon['Prob'].plot(color='purple', linewidth=2)
plt.axhline(80, linestyle='--', color='g', label='High Decile')
plt.axhline(60, linestyle='--', color='lime', label='Moderate Decile')
plt.axhline(40, linestyle='--', color='orange', label='Low Decile')
plt.axhline(20, linestyle='--', color='r', label='Very Low Decile')
plt.title('Six-Monthly Decile Index (DI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylabel('Decile Rank (%)')
plt.legend()
plt.show()


# Hutchinson Drought Index (HDI)
plt.figure(figsize=(15, 7))
df_6mon['HDI'].plot(color='green', linewidth=2)
plt.axhline(1, linestyle='--', color='g', label='Moderate Drought')
plt.axhline(-1, linestyle='--', color='r', label='Severe Drought')
plt.title('Six-Monthly Hutchinson Drought Index (HDI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylabel('HDI')
plt.legend()
plt.show()

# Z-Score Index (ZSI)
plt.figure(figsize=(15, 7))
df_6mon['ZSI'].plot(color='red', linewidth=2)
plt.axhline(1, linestyle='--', color='g', label='Above Normal')
plt.axhline(-1, linestyle='--', color='r', label='Below Normal')
plt.title('Six-Monthly Z-Score Index (ZSI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylim(-3, 3)
plt.ylabel('Z-Score')
plt.legend()
plt.show()

# China-Z Index (CZI)
plt.figure(figsize=(15, 7))
df_6mon['CZI'].plot(color='blue', linewidth=2)
plt.axhline(1, linestyle='--', color='g', label='Above Normal')
plt.axhline(-1, linestyle='--', color='r', label='Below Normal')
plt.title('Six-Monthly China-Z Index (CZI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylim(-3, 3)
plt.ylabel('CZI')
plt.legend()
plt.show()

# Modified China-Z Index (MCZI)
plt.figure(figsize=(15, 7))
df_6mon['MCZI'].plot(color='purple', linewidth=2)
plt.axhline(1, linestyle='--', color='g', label='Above Normal')
plt.axhline(-1, linestyle='--', color='r', label='Below Normal')
plt.title('Six-Monthly Modified China-Z Index (MCZI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylim(-3, 3)
plt.ylabel('MCZI')
plt.legend()
plt.show()

# Rainfall Anomaly Index (RAI)
plt.figure(figsize=(15, 7))
df_6mon['RAI'].plot(color='green', linewidth=2)
plt.axhline(0.5, linestyle='--', color='g', label='Slightly Above Average')
plt.axhline(-1, linestyle='--', color='r', label='Below Average')
plt.title('Six-Monthly Rainfall Anomaly Index (RAI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylim(-4, 4)
plt.ylabel('RAI')
plt.legend()
plt.show()

# Plotting EDI
plt.figure(figsize=(15, 7))
df_6mon['EDI'].plot(color='b', linewidth=2)
plt.axhline(0, linestyle='--', color='g', label='Average')
plt.axhline(-1, linestyle='--', color='r', label='Below Normal')
plt.axhline(1, linestyle='--', color='orange', label='Above Normal')
plt.title('Standardized Precipitation Index (EDI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylabel('EDI Value')
plt.legend()
plt.grid()
plt.show()

# Plotting SPI
plt.figure(figsize=(15, 7))
df_6mon['SPI'].plot(color='purple', linewidth=2)
plt.axhline(0, linestyle='--', color='g', label='Average')
plt.axhline(-1, linestyle='--', color='r', label='Below Normal')
plt.axhline(1, linestyle='--', color='orange', label='Above Normal')
plt.title('Standardized Precipitation Index (SPI)', fontsize=16)
plt.xlim(df_6mon.index.min(), df_6mon.index.max())
plt.ylabel('SPI Value')
plt.legend()
plt.grid()
plt.show()




nan_dict = {}

# Iterate through columns to find NaN values
for column in df_6mon.columns:
    nan_rows = df_6mon[df_6mon[column].isnull()].index.tolist()
    if nan_rows:
        nan_dict[column] = nan_rows

# Print the results
if nan_dict:
    print("Columns with NaN values and their corresponding rows:")
    for column, rows in nan_dict.items():
        print(f"Column: {column}, Rows: {rows}")
else:
    print("No NaN values found in the DataFrame.")
