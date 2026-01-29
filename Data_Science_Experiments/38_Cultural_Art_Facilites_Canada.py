import pandas as pd
from sklearn.cluster import DBSCAN
import hdbscan
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")

print(df.head())

print(df['ODCAF_Facility_Type'].value_counts())

df_museums = df[df['ODCAF_Facility_Type'] == 'museum']

print(df_museums['ODCAF_Facility_Type'].value_counts())

df = df[['Latitude','Longitude']]

print(df.head())

df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

df = df.dropna(subset=['Latitude', 'Longitude'])

print(df.dtypes)

coords_scaled = df.copy()

coords_scaled['Latitude'] = 2 * coords_scaled['Latitude']

min_samples = 3

eps = 1.0

metric = 'euclidean'

dfscan = DBSCAN(eps=eps,min_samples=min_samples,metric=metric)

dfscan.fit(coords_scaled)
df['cluster'] = dfscan.labels_

print(df['cluster'].value_counts())

min_samples=None
min_cluster_size=3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')

df['Cluster'] = hdb.fit_predict(coords_scaled)

print(df['Cluster'].value_counts())