import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
# Load the CSV file
df = pd.read_csv('../Dataset.csv')

def create_preference_vectors_given_bounds(top_users, top_items):
    # Filter data to include only the top most frequently rated items
    filtered_df = df[df['Item'].isin(top_items)]

    # Filter data to include only the top users who have rated the most items
    filtered_df = filtered_df[filtered_df['User'].isin(top_users)]

    # Now create the pivot table
    user_item_matrix = filtered_df.pivot_table(index='User', columns='Item', values='Rating', aggfunc='max')

    # Fill NaN values with 0, which represent unrated items
    user_preference_vectors = user_item_matrix.fillna(0)

    return user_preference_vectors

# user_preference_vectors = create_preference_vectors(final_df)
top_items = df['Item'].value_counts().nlargest(25000).index
top_users = df['User'].value_counts().nlargest(25000).index

user_preference_vectors = create_preference_vectors_given_bounds(top_users, top_items)

# Show the transformed DataFrame
print(user_preference_vectors.head())
# Normalize the data to unit length, necessary for cosine similarity
normalized_vectors = normalize(user_preference_vectors)

# Convert cosine similarity to distance
cosine_distances = 1 - cosine_similarity(normalized_vectors)

# Applying PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_vectors)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_vectors)

# Plotting
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_data[:, 0],
                      reduced_data[:, 1], c=clusters,
                      cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('User Clusters after PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()
