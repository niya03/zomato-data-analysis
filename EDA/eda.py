import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv("../DATASETS/zomato_cleaned.csv")
df.info()

#Top 10 most common cuisines
cuisines_series = df['cuisines'].dropna().apply(lambda x: x.split(', '))
cuisine_list = [cuisine for sublist in cuisines_series for cuisine in sublist]
cuisine_counts = Counter(cuisine_list).most_common(10)

cuisine_df = pd.DataFrame(cuisine_counts, columns=['Cuisine', 'Count'])
sns.barplot(x='Count', y='Cuisine', data=cuisine_df)
plt.title('Top 10 Most Common Cuisines in Bangalore')
plt.show()

#Distribution of Average Cost
sns.histplot(df['averagecost'], bins=30, kde=True)
plt.title('Distribution of Average Cost for Two People')
plt.xlabel('Cost (INR)')
plt.ylabel('Number of Restaurants')
plt.show()

#Distribution of Dinner Ratings
sns.histplot(df['dinner ratings'].dropna(), bins=20, kde=True)
plt.title('Distribution of Dinner Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


#Top 10 areas with most restaurants
top_areas = df['area'].value_counts().nlargest(10)
sns.barplot(x=top_areas.values, y=top_areas.index)
plt.title('Top 10 Areas with Most Restaurants')
plt.xlabel('Number of Restaurants')
plt.ylabel('Area')
plt.show()