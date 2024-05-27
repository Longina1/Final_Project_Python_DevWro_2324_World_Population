import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('world_population.csv')
print(df.head(50).to_string())
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df.isna().sum())

print(df.describe().to_string())

print(df.groupby('Continent')['2022 Population'].mean().sort_values())

# df1 = df.iloc[41, :]
# print(df1)


print(df.head(5).to_string())


#Top 10 most populous countries
#top_ten = df.nlargest(10, '2022 Population')
top_ten = df.sort_values('2022 Population', ascending=False).head(10)
print(top_ten.to_string())
plt.figure(figsize=(10, 6))
plt.bar(top_ten['Country/Territory'], top_ten['2022 Population'])
plt.xlabel('Country/Territory')
plt.ylabel('2022 Population')
plt.title('Top ten most populous countries (2022)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

sns.barplot(x='2022 Population', y='Country/Territory', data=top_ten, hue='Country/Territory')
plt.title('Top ten most populous countries (2022)')
plt.xlabel('Population')
plt.ylabel('Country/Territory')
plt.show()


#Continents by growth rate
sns.barplot(x='Growth Rate', y='Country/Territory', data=df)
plt.title('Growth rate by country')
plt.xlabel('Growth Rate')
plt.ylabel('Continent')
plt.show()

#Population density
# sns.scatterplot(x='2022 Population', y='Density (per km²)', data=df)
# plt.title('Population density')
# plt.xlabel('2022 Population')
# plt.ylabel('Density (per km²)')
# plt.show()

#Comparing the population of Poland and Spain
poland_population = df.loc[df['Country/Territory']=='Poland', '2022 Population'].values[0]
spain_population = df.loc[df['Country/Territory']=='Spain', '2022 Population'].values[0]

plt.figure(figsize=(10, 6))
plt.bar(['Poland', 'Spain'], [poland_population, spain_population])
plt.xlabel('Country/Territory')
plt.ylabel('2022 Population')
plt.title('Population comparison: Poland vs. Spain')
plt.show()

