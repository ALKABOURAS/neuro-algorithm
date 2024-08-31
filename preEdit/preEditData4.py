import pandas as pd
# Load the CSV file
df = pd.read_csv('../Dataset.csv')

# Filter data to include only the top n most frequently rated items
top_items = df['Item'].value_counts().nlargest(30000).index
filtered_df = df[df['Item'].isin(top_items)]

# Filter data to include only the top n users who have rated the most items
top_users = df['User'].value_counts().nlargest(30000).index
filtered_df = filtered_df[filtered_df['User'].isin(top_users)]
# Now create the pivot table
user_item_matrix = filtered_df.pivot_table(index='User', columns='Item', values='Rating', aggfunc='max')
# Fill NaN values with 0, which represent unrated items
user_preference_vectors = user_item_matrix.fillna(0)

# Show the transformed DataFrame
print(user_preference_vectors.head())
