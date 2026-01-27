import pandas as pd
from IPython.core.pylabtools import figsize
from matplotlib.pyplot import xticks
from scipy.stats import alpha

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

import numpy as np

import matplotlib.pyplot as plt

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

print('Data read into a pandas dataframe!')

df_can.set_index('Country',inplace=True)

df_can.index.name = None

print(df_can.head())

years = list(map(str,range(1980,2014)))

#Area Plots

df_can.sort_values(by='Total',ascending=False,axis=0,inplace=True)

df_top5 = df_can.head(5)

df_top5 = df_top5[years].transpose()

print(df_top5)

df_top5.index = df_top5.index.map(int)

df_top5.plot(kind= 'area',alpha=0.25,stacked = True,figsize = (20,10))

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

#bottom 5 area plot

df_can.sort_values(by="Total",ascending=True,axis=0,inplace=True)

df_bottom5 = df_can.head()

df_bottom5 = df_bottom5[years].transpose()

df_bottom5.index = df_bottom5.index.map(int)

df_bottom5.plot(kind='area',alpha=0.55,stacked = False,figsize = (10,10))

plt.title('Immigration Trend of Botttom 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

#Histograms

# print(df_can['2013'].head())

count, bin_edges = np.histogram(df_can['2013'])

print(count)
print(bin_edges)

df_can['2013'].plot(kind = 'hist',figsize = (8,5),xticks=bin_edges)

plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

plt.show()

df_t = df_can.loc[['Denmark','Sweden','Norway'],years].transpose()

df_t.plot(kind = 'hist',figsize = (10,6))

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

count , bin_edges = np.histogram(df_t,15)

print(count)
print(bin_edges)

df_t.plot(kind= 'hist',figsize = (10,6),bins = 15,alpha= 0.6,xticks = bin_edges)

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

# stacked Histogram
df_t.plot(kind='hist',
          figsize=(10, 6),
          bins=15,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=False,
          xlim=(xmin, xmax)
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

df_cof = df_can.loc[['Greece','Albania','Bulgaria'],years]

df_cof = df_cof.transpose()

count,bin_edges = np.histogram(df_cof,15)

df_cof.plot(kind='hist',figsize = (10,6),bins = 15,alpha=0.35,xticks = bin_edges,color=['coral', 'darkslateblue', 'mediumseagreen'])

plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

#Bar charts

df_iceland = df_can.loc["Iceland",years]

print(df_iceland.head())

df_iceland.plot(kind = 'bar',figsize = (10,6))

plt.xlabel('Year')
plt.ylabel('Number of immigrants')
plt.title('Icelandic immigrants to Canada from 1980 to 2013')

plt.annotate('',  # s: str. Will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)

             )
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=72.5,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned.
             )
plt.show()


df_can.sort_values(by="Total",ascending=False,inplace=True)

df_top15 = df_can['Total'].head(15)

print(df_top15)

df_top15.plot(kind = 'barh', figsize=(10,6))

plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')

plt.show()
