import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

print('Data read into a pandas dataframe!')

df_can.set_index('Country',inplace=True)

print(df_can.head())

years = list(map(str,range(1980,2014)))

df_continents = df_can.groupby('Continent',axis=0).sum()

print(df_continents)

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

df_continents['Total'].plot(kind='pie',
                            figsize=(10, 6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            colors=colors_list,
                            explode=explode_list
                            )


plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12, fontsize = 15)

plt.axis('equal')

# add legend
plt.legend(labels=df_continents.index, loc='upper left', fontsize=7)

plt.show()

explode_list = [0.0, 0, 0, 0.1, 0.1, 0.2]

df_continents['2013'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            explode=explode_list
                            )

plt.title('Immigration to Canada by Continent in 2013', y=1.12)
plt.axis('equal')


plt.legend(labels=df_continents.index, loc='upper left')


plt.show()

df_japan = df_can.loc[['Japan'],years].transpose()

df_japan.plot(kind='box',figsize=(8,6))

plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')

plt.show()

#Box plot
df_CI =  df_can.loc[['China','India'],years].transpose()

print(df_CI.head())

print(df_CI.describe())

df_CI.plot(kind='box',figsize= (10,6))

plt.title('Box plot of Chinese and Indian Immigrants from 1980 - 2013')

plt.ylabel('Number of Immigrants')

plt.show()

#Subplots

fig = plt.figure()
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)

df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.show()
