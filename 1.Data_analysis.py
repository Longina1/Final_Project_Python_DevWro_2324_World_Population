import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('world_population.csv')

print(df.head().to_string())
print(df.describe().to_string())

print(df.info())
# print(df.shape)
print(df.isna().sum())


print(df.groupby('Continent')['2022 Population'].mean().sort_values())


#Top 10 most populous countries
top_ten = df.sort_values('2022 Population', ascending=False).head(10)
print(top_ten.to_string())


plt.figure(figsize=(10, 6))
sns.barplot(x='2022 Population', y='Country/Territory', data=top_ten, hue='Country/Territory')
plt.title('Top ten most populous countries (2022)')
plt.xlabel('Population')
plt.ylabel('Country/Territory')
plt.show()


#Comparing the population of Poland and Spain
poland_population = df.loc[df['Country/Territory']=='Poland', '2022 Population'].values[0]
spain_population = df.loc[df['Country/Territory']=='Spain', '2022 Population'].values[0]

plt.figure(figsize=(10, 6))
plt.bar(['Poland', 'Spain'], [poland_population, spain_population], color='lightblue')
plt.xlabel('Country/Territory')
plt.ylabel('2022 Population')
plt.title('Population comparison: Poland vs. Spain')
plt.show()

