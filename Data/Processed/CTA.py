#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.fft import fft, ifft, fftfreq, rfft, rfftfreq
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#import seaborn as sns 
#import datetime

#%%

#%%
#Read the CSV Files
df = pd.read_csv('Data/Raw_CTA/CTA_Data_File.csv', parse_dates=['date'])
df['rides'] = df['rides'].str.replace(",", "", regex=False).astype(float)
#Group by the station name and sum up the rides to get a list of biggest to smallest
station_totals = df.groupby('stationname')['rides'].sum().sort_values(ascending = False)

#Get the top 10, I dont need every single station
Top_10_Stations = station_totals.head(10)
#Now mask df to only get where station name is contained in the top 10 stations
#Then only keep useful columns and put the index as the date
CTA_Top_10_df = df[df['stationname'].isin(Top_10_Stations.index)][['date', 'stationname', 'rides', 'daytype']].set_index(['date'])
#Get the data from 2010 and later just because it is too much data
CTA_Top_10_df = CTA_Top_10_df[CTA_Top_10_df.index >= '2010-01-01'] 
#%%
#plt.figure(figsize=(12,5))
##array = np.asarray(Addison_North_Main['rides'])
#array = np.ascontiguousarray(Addison_North_Main['rides'])
#yf = rfft(array)
#xf = rfftfreq(len(array), 1)
#plt.plot(xf, np.abs(yf))
#plt.xlabel('Frequency (Cycles/Day)')
#plt.ylabel('Ampltitude')
#plt.xlim(-0.001,0.01)

