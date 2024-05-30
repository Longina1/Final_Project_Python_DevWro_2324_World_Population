import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def predict_population(country):
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
            'Density (per km²)', 'Growth Rate', 'World Population Percentage'],axis=1, inplace=True)
    country_population = country_population.T
    country_population = country_population.reset_index().set_axis(['Year', 'Population'], axis='columns')
    country_population.drop(0, axis=0, inplace=True)

    X = country_population.iloc[:, 0].values.reshape(-1, 1)
    y = country_population.iloc[:, 1].values.reshape(-1, 1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"The linear regression model is applied to {country}'s population data.")
    print(f'The accuracy of the model is {model.score(X_test, y_test)}.')


# predict_population('China')
predict_population('India')
# predict_population('United States')