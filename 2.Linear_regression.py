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
print(df.head().to_string())

china_population = df.loc[df['Country/Territory']=='China']
china_population.drop(['Rank','CCA3','Capital','Continent', 'Area (km²)',
            'Density (per km²)', 'Growth Rate', 'World Population Percentage'],axis=1, inplace=True)
print(china_population)

china_population = china_population.T
print(china_population.head(8))

china_population = china_population.reset_index().set_axis(['Year', 'Population'], axis='columns')
china_population.drop(0, axis=0, inplace=True)
print(china_population.head(8))

print(china_population.iloc[:, 0].ndim)
print(china_population.iloc[:, 0].shape)


X = china_population.iloc[:, 0].values.reshape(-1, 1)
y = china_population.iloc[:, 1].values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)


print(model.score(X_test, y_test))


#Persisting the model
joblib.dump(model, 'Linear_regression.model')


#Loading the model
#model_1 = joblib.load('Linear_regression.model')
