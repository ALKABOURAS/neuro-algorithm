import pandas as pd


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

def jaccard_distance(set_a, set_b):
    # Compute the intersection and union of the two sets
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    # Compute the Jaccard distance
    distance = 1 - intersection / union

    return distance

def compute_all_jaccard_distances(user_preference_vectors):
    # Convert the DataFrame into a dictionary of sets
    user_sets = {user_id: set(user_data[user_data > 0].index)
                 for user_id, user_data in user_preference_vectors.iterrows()}

    # Initialize an empty dictionary to store distances
    distances = {}

    # Compute Jaccard distances for all pairs of users
    for user_id_u, items_u in user_sets.items():
        for user_id_v, items_v in user_sets.items():
            if user_id_u != user_id_v:  # Avoid comparing a user with themselves
                distance = jaccard_distance(items_u, items_v)
                distances[(user_id_u, user_id_v)] = distance

    return distances


top_items = df['Item'].value_counts().nlargest(5000).index
top_users = df['User'].value_counts().nlargest(5000).index

user_preference_vectors = create_preference_vectors_given_bounds(top_users, top_items)

# Show the transformed DataFrame
print(user_preference_vectors.head())

jaccard_distances = compute_all_jaccard_distances(user_preference_vectors)

# Print the Jaccard distances for the first 5 pairs of users

for (user_id_u, user_id_v), distance in list(jaccard_distances.items())[:5]:
    print(f"Jaccard distance between User {user_id_u} and User {user_id_v}: {distance}")