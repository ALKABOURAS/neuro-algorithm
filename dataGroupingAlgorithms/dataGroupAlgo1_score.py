import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the CSV file
from dataClass import DataProcessor


data = DataProcessor('../Dataset.csv')

user_preference_vectors = data.create_preference_vectors_given_bounds(2000, 2000)


# Show the transformed DataFrame
print(user_preference_vectors.head())

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(user_preference_vectors)

# Find the optimal number of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_data)
    score = silhouette_score(normalized_data, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores
import matplotlib.pyplot as plt
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.show()
