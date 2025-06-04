import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("DATASETS/zomato_cleaned.csv")

# Features for Clustering
features = [
    'averagecost', 'ishomedelivery', 'istakeaway', 'isvegonly',
    'isindoorseating', 'area', 'cuisines'
]
X = df[features].copy(deep=True)

# Fill missing values
X['averagecost'] = X['averagecost'].fillna(X['averagecost'].median())
X['area'] = X['area'].fillna("Unknown")
X['cuisines'] = X['cuisines'].fillna("Other")

# Use only the first listed cuisine
X['cuisines'] = X['cuisines'].apply(lambda x: x.split(',')[0].strip())

# Define feature types
numeric_features = ['averagecost']
categorical_features = ['area', 'cuisines']
binary_features = ['ishomedelivery', 'istakeaway', 'isvegonly', 'isindoorseating']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Keep binary features as-is
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# KMeans Clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

# Assign cluster labels to original DataFrame
df['cluster'] = clusters

# PCA for Visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=50)
plt.title("Zomato Restaurant Clusters (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Cluster Summary
print("\nCluster Summary Statistics:")
cluster_summary = df.groupby('cluster').agg({
    'averagecost': 'mean',
    'ishomedelivery': 'mean',
    'istakeaway': 'mean',
    'isvegonly': 'mean',
    'isindoorseating': 'mean'
})
cluster_summary['restaurant_count'] = df.groupby('cluster').size()
print(cluster_summary)

# Plot Feature Profiles per Cluster
features_to_plot = ['averagecost', 'ishomedelivery', 'istakeaway', 'isvegonly', 'isindoorseating']
cluster_summary[features_to_plot].plot(kind='bar', figsize=(12,10), subplots=True, layout=(2,3), legend=False)
plt.suptitle('Cluster Feature Profiles', fontsize=16)
plt.show()

# Top Cuisines per Cluster
for i in range(k):
    top_cuisines = df[df['cluster'] == i]['cuisines'].value_counts().head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
    plt.title(f'Top 5 Cuisines in Cluster {i}')
    plt.xlabel('Number of Restaurants')
    plt.ylabel('Cuisine')
    plt.show()

# Export Clustered Dataset
columns_to_export = ['name', 'cuisines', 'area', 'averagecost','cluster']
existing_columns = [col for col in columns_to_export if col in df.columns]
df_export = df[existing_columns]
df_export.to_csv('DATASETS/zomato_cluster.csv', index=False)
