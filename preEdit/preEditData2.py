import pandas as pd

# Load the CSV file
df = pd.read_csv('../Dataset.csv')

def filter_data(df, R_min, R_max):
    # Count the number of ratings per user
    user_rating_counts = df['User'].value_counts()

    # Filter users based on R_min and R_max
    filtered_users = user_rating_counts[(user_rating_counts >= R_min) &
                                        (user_rating_counts <= R_max)].index

    # Apply the filter to the dataset
    filtered_df = df[df['User'].isin(filtered_users)]

    # Now filter items that are only rated by the filtered users
    filtered_items = filtered_df['Item'].value_counts().index

    # Final filtered DataFrame
    final_df = filtered_df[filtered_df['Item'].isin(filtered_items)]

    return final_df, filtered_users.size, filtered_items.size

# Usage
final_df, num_users, num_items = filter_data(df, 5, 9)
print("Number of ratings per user:", num_users)
print("Number of items:", num_items)
print(final_df.head())