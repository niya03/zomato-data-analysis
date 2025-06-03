import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("zomato_cleaned.csv")

#Features for Clustering
features = [
    'averagecost', 'ishomedelivery', 'istakeaway', 'isvegonly',
    'isindoorseating', 'area', 'cuisines']

X = df[features].copy()

#Data Preprocessing

#Fill missing values
X['averagecost'] = X['averagecost'].fillna(X['averagecost'].median())
X['area'] = X['area'].fillna("Unknown")
X['cuisines'] = X['cuisines'].fillna("Other")

X['cuisines'] = X['cuisines'].apply(lambda x: x.split(',')[0].strip())

#feature types
numeric_features = ['averagecost']
categorical_features = ['area', 'cuisines']
binary_features = ['ishomedelivery', 'istakeaway', 'isvegonly', 'isindoorseating']

#Preprocessing pipeline:

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # keep binary features unchanged
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

#KMeans Clustering

k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_processed)
df['cluster'] = clusters

#Visualize Clusters with PCA

# Reduce dimensionality for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=50)
plt.title("Zomato Restaurant Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

#Cluster Interpretation

print("\nCluster Summary Statistics:")
cluster_summary = df.groupby('cluster').agg({
    'averagecost': 'mean',
    'ishomedelivery': 'mean',
    'istakeaway': 'mean',
    'isvegonly': 'mean',
    'isindoorseating': 'mean',
    'name': 'count'  # number of restaurants per cluster
}).rename(columns={'name': 'restaurant_count'})

print(cluster_summary)

features_to_plot = ['averagecost', 'ishomedelivery', 'istakeaway', 'isvegonly', 'isindoorseating']

# bar charts for each feature per cluster
cluster_summary[features_to_plot].plot(kind='bar', figsize=(12,10), subplots=True, layout=(2,3), legend=False)
plt.suptitle('Cluster Feature Profiles', fontsize=16)
plt.show()
#top cuisines per cluster
for i in range(k):
    top_cuisines = df[df['cluster'] == i]['cuisines'].value_counts().head(5)
    plt.figure(figsize=(10,8))
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
    plt.title(f'Top 5 Cuisines in Cluster {i}')
    plt.xlabel('Number of Restaurants')
    plt.ylabel('Cuisine')
    plt.show()
