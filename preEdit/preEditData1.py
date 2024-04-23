import pandas as pd

# Load the CSV file
df = pd.read_csv('../Dataset.csv')

# Display the first 5 rows of the dataframe

# print(df.head())


def find_unique(df):
    unique_users = df['User'].unique()
    unique_items = df['Item'].unique()
    return unique_users, unique_items


unique_users, unique_items = find_unique(df)
print("Unique Users: ", unique_users)
print("Unique Items: ", unique_items)
print("Number of unique users: ", len(unique_users))
print("Number of unique items: ", len(unique_items))
