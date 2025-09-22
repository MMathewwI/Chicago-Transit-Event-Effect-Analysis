#%%
from Data.Processed.CTA import CTA_Top_10_df
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from Data.Processed.Baseball import Baseball_Table
from datetime import timedelta
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
#%%

CHA_games = Baseball_Table[Baseball_Table['hometeam'] == 'CHA']
CHN_games = Baseball_Table[Baseball_Table['hometeam'] == 'CHN']
cubs_dates = CHN_games.index
whitesocks_dates = CHA_games.index
both_dates = cubs_dates.union(whitesocks_dates)
#%%
Station_df = pd.DataFrame(columns=['stationname', 'normalavg', 'cubsavg', 'whiteavg'])
Station_df['stationname'] = CTA_Top_10_df['stationname'].unique()
for station in CTA_Top_10_df['stationname'].unique():
    station_data = CTA_Top_10_df[CTA_Top_10_df['stationname'] == station]
    #Try to isolate cub dates
    cubs_data = station_data.loc[station_data.index.isin(cubs_dates)]
    cubs_data_sum = cubs_data['rides'].sum()
    Station_df.loc[Station_df['stationname'] == station, 'cubsavg'] = (cubs_data_sum / len(cubs_data.index))
    #Try to isolate cub dates
    whitesocks_data = station_data.loc[station_data.index.isin(whitesocks_dates)]
    whitesocks_data_sum = cubs_data['rides'].sum()
    Station_df.loc[Station_df['stationname'] == station, 'whiteavg'] = (whitesocks_data_sum / len(whitesocks_data.index))
    nonbaseball_data = station_data.loc[~station_data.index.isin(both_dates)]
    nonbaseball_data_sum = nonbaseball_data['rides'].sum()
    Station_df.loc[Station_df['stationname'] == station, 'normalavg'] = (nonbaseball_data_sum / len(nonbaseball_data.index))
    Station_df['cubs_percent_diff'] = (Station_df['cubsavg'] - Station_df['normalavg']) / Station_df['normalavg']
    Station_df['whitesocks_percentage_diff'] = (Station_df['whiteavg'] - Station_df['normalavg']) / Station_df['normalavg']
Station_df = Station_df.set_index('stationname')
x = np.arange(len(Station_df))
plt.bar(x - 0.2, Station_df['cubs_percent_diff'], 0.4, label = 'Cubs', color = 'red')
plt.bar(x + 0.2, Station_df['whitesocks_percentage_diff'], 0.4, label = 'White Socks', color = 'gray')
plt.xticks(x, Station_df.index, rotation=90)
plt.title('Percent Difference between Ridership during Normal days and Baseball days')
plt.ylabel('Percentage Difference')
plt.legend()
plt.xlabel('Name of Rides')
plt.show()

#%%

#Plot every station and the rides
for station in CTA_Top_10_df['stationname'].unique():
    #Set up figure
    plt.figure(figsize=(18,5))
    #Binary mask is where the station name is the station being indexed
    binary_mask = CTA_Top_10_df['stationname'] == station
    station_data = CTA_Top_10_df[binary_mask]
    Cubs_Games = station_data.loc[station_data.index.isin(cubs_dates), 'rides']    
    Whitesox_Games = station_data.loc[station_data.index.isin(whitesocks_dates), 'rides']
    Fit_Model = ExponentialSmoothing(station_data['rides'], trend="add", seasonal="add", seasonal_periods=365).fit()
    #Forecast the Fit 5 years
    Model_forecast = Fit_Model.forecast(1825)
    last_date = station_data.index[-1]
    Model_forecast.index = pd.date_range(start = last_date, periods = 1825)
    #Plot it all
    plt.scatter(station_data.index, station_data['rides'], label = 'Original Data', color = 'black', s= 0.4)
    plt.plot(station_data.index, Fit_Model.fittedvalues, linestyle="-", label = 'Fitted Line', color = 'red', alpha = 0.3)
    plt.plot(Model_forecast.index, Model_forecast, label = '5 Year Forcecast', alpha = 0.5)
    plt.scatter(Cubs_Games.index, Cubs_Games, color = 'lime', s= 0.4, label = 'Cubs Games')
    plt.scatter(Whitesox_Games.index, Whitesox_Games, color='darkorange', s=0.4, label='White Sox Games')
    plt.legend()
    plt.title('Daily Rides for ' + station + ' (2010+)')
    plt.ylabel("# of Rides")
    plt.xlabel('Date (Years)')
    plt.xticks(rotation=90)

plt.show()

# %%
for station in CTA_Top_10_df['stationname'].unique():
    plt.figure(figsize=(18,5))
    binary_mask = CTA_Top_10_df['stationname'] == station
    station_data = CTA_Top_10_df[binary_mask].interpolate(method='linear')
    station_data['is_cubs_game'] = station_data.index.isin(cubs_dates).astype(int)
    station_data['is_whitesox_game'] = station_data.index.isin(whitesocks_dates).astype(int)
    exog = station_data[['is_cubs_game', 'is_whitesox_game']]
    if (station == 'Chicago/State'):
        model = SARIMAX(station_data['rides'],order=(1,1,1),seasonal_order=(1,1,1,7),exog=exog)
    else:
        model = SARIMAX(station_data['rides'],order=(2,1,2),seasonal_order=(1,1,1,7),exog=exog)
    model_fit = model.fit()

    plt.scatter(station_data.index, station_data['rides'], color = 'black', s= 0.4, label = 'Original Data')
    plt.scatter(Cubs_Games.index, Cubs_Games, color = 'lime', s= 0.4, label = 'Cubs Games')
    plt.scatter(Whitesox_Games.index, Whitesox_Games, color='darkorange', s=0.4, label='White Sox Games')
    plt.plot(station_data.index, model_fit.fittedvalues, linestyle="-", label = 'Fitted Line', color = 'red', alpha = 0.3)
    plt.plot(Model_forecast.index, Model_forecast, label = '5 Year Forcecast', alpha = 0.5)
    plt.legend()
    plt.title('Daily Rides for ' + station + ' (2010+)')
    plt.ylabel("# of Rides")
    plt.xlabel('Date (Years)')
    plt.xticks(rotation=90)
