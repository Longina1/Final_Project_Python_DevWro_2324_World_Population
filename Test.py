import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv('world_population.csv')

df = df.rename(columns={'2022 Population': '2022',
                        '2020 Population': '2020',
                        '2015 Population': '2015',
                        '2010 Population': '2010',
                        '2000 Population': '2000',
                        '1990 Population': '1990',
                        '1980 Population': '1980',
                        '1970 Population': '1970'})
print(df.head(5).to_string())

china_population = df.iloc[41, :]
print(china_population)
print(china_population.shape)
# china_population.T
china_population.drop('Rank', 'CCA3', 'Capital', 'Continent', 'Area (km²)', 'Density (per km²)', 'Growth Rate',
                      'World Population Percentage', axis=0, inplace=True)
print(china_population)

# china_population.drop(['Rank','CCA3','Capital','Continent', 'Area (km²)',
#             'Density (per km²)', 'Growth Rate', 'World Population Percentage'],axis=1, inplace=True) #1 is the axis number (0 for rows and 1 for columns.)

# china_population = china_population.T
# print(china_population.head(8))