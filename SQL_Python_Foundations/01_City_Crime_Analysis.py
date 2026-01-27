import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

conn = sqlite3.connect("Hyderabad_City_Data.db")
cursor = conn.cursor()


census_data = {
    'Community_Area_Number': [1, 2, 3, 4, 5],
    'Community_Area_Name': ['Banjara Hills', 'Hitech City', 'Kukatpally', 'Secunderabad', 'Old City (Charminar)'],
    'Hardship_Index': [10, 15, 45, 55, 90],
    'Per_Capita_Income': [150000, 140000, 60000, 50000, 25000]
}
df_census = pd.DataFrame(census_data)
df_census.to_sql('CENSUS_DATA', conn, if_exists='replace', index=False)
#print(df_census.head())

schools_data = {
    'School_ID': [501, 502, 503, 504, 505],
    'Name_of_School': ['Hyderabad Public School', 'Oakridge Int', 'Kukatpally Govt High', 'Kendriya Vidyalaya Sec', 'Govt Urdu School Charminar'],
    'Safety_Score': [95, 98, 60, 75, 40],
    'Community_Area_Number': [1, 2, 3, 4, 5] # Links to the localities above
}
df_schools = pd.DataFrame(schools_data)
df_schools.to_sql('HYDERABAD_SCHOOLS', conn, if_exists='replace', index=False)
#print(df_schools.head())

crime_data = {
    'ID': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
    'Primary_Type': ['THEFT', 'CYBER FRAUD', 'THEFT', 'TRAFFIC VIOLATION', 'THEFT', 'ASSAULT', 'CYBER FRAUD'],
    'Community_Area_Number': [4, 2, 5, 1, 3, 5, 2]
}
df_crime = pd.DataFrame(crime_data)
df_crime.to_sql('HYDERABAD_CRIME_DATA', conn, if_exists='replace', index=False)
#print(df_crime.head())

query_census = """SELECT * FROM CENSUS_DATA"""

df_census_data = pd.read_sql_query(query_census,conn)

#print(df_census_data)

query_school_census = """SELECT Community_Area_Name,Hardship_Index,Name_Of_School,Safety_Score FROM CENSUS_DATA AS cd JOIN  HYDERABAD_SCHOOLS AS hs ON cd.COMMUNITY_AREA_NUMBER = hs.COMMUNITY_AREA_NUMBER"""

df_school_census = pd.read_sql_query(query_school_census,conn)

#print(df_school_census)

query_master = """SELECT cd.Community_Area_Name, cd.Hardship_Index,hs.Name_of_School, hcd.Primary_Type \
                  FROM CENSUS_DATA AS cd JOIN HYDERABAD_SCHOOLS as hs \
                  ON cd.COMMUNITY_AREA_NUMBER = hs.COMMUNITY_AREA_NUMBER \
                  JOIN HYDERABAD_CRIME_DATA AS hcd ON \
                  hs.COMMUNITY_AREA_NUMBER = hcd.COMMUNITY_AREA_NUMBER"""

Three_Tables_data_df = pd.read_sql_query(query_master,conn)

#print(Three_Tables_data_df)

query_crime_area = """SELECT cd.Community_Area_Name , COUNT(hcd.ID) AS TOTAL_CRIMES FROM \
                     CENSUS_DATA as cd JOIN HYDERABAD_CRIME_DATA as hcd ON \
                     cd.COMMUNITY_AREA_NUMBER = hcd.COMMUNITY_AREA_NUMBER \
                     GROUP BY cd.Community_Area_Name ORDER BY COUNT(hcd.ID)  DESC """


df_crime_area = pd.read_sql_query(query_crime_area,conn)

print(df_crime_area)