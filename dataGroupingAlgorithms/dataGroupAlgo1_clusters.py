import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
# Load the CSV file
df = pd.read_csv('../Dataset.csv')

def filter_data(df, R_min, R_max):
    # Count the number of ratings per user
    user_rating_counts = df['User'].value_counts()

    # Filter users based on R_min and R_max
    filtered_users = user_rating_counts[(user_rating_counts >= R_min) & (user_rating_counts <= R_max)].index

    # Apply the filter to the dataset
    filtered_df = df[df['User'].isin(filtered_users)]

    # Now filter items that are only rated by the filtered users
    filtered_items = filtered_df['Item'].value_counts().index

    # Final filtered DataFrame
    final_df = filtered_df[filtered_df['Item'].isin(filtered_items)]

    return final_df, filtered_users.size, filtered_items.size

def create_preference_vectors(df):
    # Create the user-item matrix
    user_item_matrix = df.pivot_table(index='User', columns='Item', values='Rating', aggfunc='max')

    # Fill NaN values with 0, which represent unrated items
    user_preference_vectors = user_item_matrix.fillna(0)

    return user_preference_vectors

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

# Usage
final_df, num_users, num_items = filter_data(df, 5, 15)
print("Number of ratings per user:", num_users)
print("Number of items:", num_items)

# user_preference_vectors = create_preference_vectors(final_df)
top_items = df['Item'].value_counts().nlargest(1000).index
top_users = df['User'].value_counts().nlargest(1000).index

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
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('User Clusters after PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()
