from unittest.mock import inplace

import pandas as pd
from IPython.core.pylabtools import figsize

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib as mpl
import matplotlib.pyplot as plt

df_can = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv')

print('Data read into a pandas dataframe!')

#print(df_can.head())

df_can.set_index("Country",inplace=True)

df_can.index.name = None

print(df_can.head(3))

years = list(map(str,range(1980,2014)))

print(mpl.__version__)

print(plt.style.available)

mpl.style.use(["ggplot"])

#Line plots

haiti  = df_can.loc['Haiti',years]

print(haiti.head())

haiti.index =  haiti.index.map(int)

# haiti.plot(kind = "line")
#
# plt.title("Immigration from Haiti")
# plt.xlabel("Years")
# plt.ylabel("Number of immigrants")
#
# plt.text(2000,6000,"2010 Earthquake")
#
# plt.show()


#India China comparision

df_IC = df_can.loc[['India','China'],years]

# print(df_IC.head())
df_IC = df_IC.transpose()

# df_IC.index = df_IC.index.map(int)
#
# df_IC.plot(kind = 'line')
#
# plt.title('Immigrants from China and India')
# plt.ylabel('Number of Immigrants')
# plt.xlabel('Years')
#
# plt.show()

#Top 5 countries that contributed the most

df_can.sort_values(by='Total',ascending=False,axis=0,inplace=True)

df_top5 = df_can.head(5)

df_top5 = df_top5[years].transpose()

df_top5.index = df_top5.index.map(int)

print(df_top5)

df_top5.plot(kind='line',figsize= (14,8))

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')


plt.show()


