# Task 8: K-Means Clustering - Mall Customer Segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Step 1: Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Drop non-numeric columns
df = df.drop(['CustomerID', 'Genre'], axis=1)

# Step 3: Visualize Feature Distributions (optional)
sns.pairplot(df)
plt.suptitle("Feature Distributions", y=1.02)
plt.show()

# Step 4: Elbow Method to choose optimal K
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, inertia, 'bo-')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.show()

# Step 5: Train K-Means with optimal K (e.g., 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Step 6: Evaluate using Silhouette Score
score = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])
print(f"✅ Silhouette Score: {score:.2f}")

# Step 7: Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.drop('Cluster', axis=1))
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Step 8: Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100)
plt.title("Customer Segments Visualized via PCA (K-Means Clustering)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
