import pandas as pd


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

def jaccard_distance(set_a, set_b):
    # Compute the intersection and union of the two sets
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    # Compute the Jaccard distance
    distance = 1 - intersection / union

    return distance

def compute_all_jaccard_distances(user_preference_vectors):
    # Convert the DataFrame into a dictionary of sets
    user_sets = {user_id: set(user_data[user_data > 0].index) for user_id, user_data in user_preference_vectors.iterrows()}

    # Initialize an empty dictionary to store distances
    distances = {}

    # Compute Jaccard distances for all pairs of users
    for user_id_u, items_u in user_sets.items():
        for user_id_v, items_v in user_sets.items():
            if user_id_u != user_id_v:  # Avoid comparing a user with themselves
                distance = jaccard_distance(items_u, items_v)
                distances[(user_id_u, user_id_v)] = distance

    return distances


final_df, num_users, num_items = filter_data(df, 5, 15)
print("Number of ratings per user:", num_users)
print("Number of items:", num_items)

# user_preference_vectors = create_preference_vectors(final_df)
top_items = df['Item'].value_counts().nlargest(2000).index
top_users = df['User'].value_counts().nlargest(2000).index

user_preference_vectors = create_preference_vectors_given_bounds(top_users, top_items)

# Show the transformed DataFrame
print(user_preference_vectors.head())

jaccard_distances = compute_all_jaccard_distances(user_preference_vectors)

# Print the Jaccard distances for the first 5 pairs of users

for (user_id_u, user_id_v), distance in list(jaccard_distances.items())[:5]:
    print(f"Jaccard distance between User {user_id_u} and User {user_id_v}: {distance}")