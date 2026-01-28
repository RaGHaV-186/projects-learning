import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

print(cust_df.head())

cust_df = cust_df.drop('Address',axis=1)

cust_df =  cust_df.dropna()

print(cust_df.info())

X = cust_df.values[:,1:]

Clus_dataSet = StandardScaler().fit_transform(X)

clustNum = 3

k_means = KMeans(init="k-means++",n_clusters=clustNum,n_init=12)

k_means.fit(X)

labels = k_means.labels_

cust_df['Clus_km'] = labels

print(cust_df.head())

print(cust_df.groupby('Clus_km').mean())

area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))

fig.show()

cust_df_sub = cust_df[['Age', 'Edu','Income','Clus_km']].copy()
sns.pairplot(cust_df_sub, hue='Clus_km', palette='viridis', diag_kind='kde')
plt.suptitle('Pairwise Scatter Plot with K-means Clusters', y=1.02)
plt.show()


