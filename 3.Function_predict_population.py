import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def predict_population(country, year):
    df = pd.read_csv('world_population.csv')
    df = df.rename(columns={'2022 Population': '2022',
                        '2020 Population': '2020',
                        '2015 Population': '2015',
                        '2010 Population': '2010',
                        '2000 Population': '2000',
                        '1990 Population': '1990',
                        '1980 Population': '1980',
                        '1970 Population': '1970'})


    country_population = df.loc[df['Country/Territory']==country]
    country_population.drop(['Rank','CCA3','Capital','Continent', 'Area (km²)',
            'Density (per km²)', 'Growth Rate', 'World Population Percentage'],axis=1, inplace=True) #1 is the axis number (0 for rows and 1 for columns.)
    country_population = country_population.T
    country_population = country_population.reset_index()
    #china_population.drop([0])
    print(country_population.head(8))

    X = country_population.iloc[1:8, 0].values.reshape(-1, 1)
    y = country_population.iloc[1:8, 1].values.reshape(-1, 1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict([[year]]) #y_predict = regressor.predict(X_test)
    print(f'The population of {country} in {year} is estimated at {y_pred}.')
    print(f'The accuracy of the model is {model.score(X_test, y_test)}.')


# predict_population('China', 2019)
predict_population('India', 2019)
# predict_population('United States', 2030)