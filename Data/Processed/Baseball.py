#%%
import pandas as pd
import glob
import os

#%%
#Read all the baseball CSV files
Baseball_files = glob.glob(os.path.join('Data/External_Factors/Baseball_Data', '*.csv'))

#Make a list to append to for later
list = []

#Look through every Baseball csv file 
for file in Baseball_files:
    #Create a dataframe for each year
    df = pd.read_csv(file)
    #Format the time in the date section
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    #Find entries where it is CHA = Cubs, or CHN = White Socks
    df = df[df['hometeam'].isin(['CHA', 'CHN'])][['date', 'hometeam']]
    #Append them to my list
    list.append(df)
#Make another df
Baseball_Table = pd.concat(list)
#Set date as index
Baseball_Table = Baseball_Table.set_index("date")