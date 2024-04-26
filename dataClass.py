import pandas as pd
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def find_unique(self):
        unique_users = self.df['User'].unique()
        unique_items = self.df['Item'].unique()
        return unique_users, unique_items

    def filter_data(self, R_min, R_max):
        user_rating_counts = self.df['User'].value_counts()
        filtered_users = user_rating_counts[(user_rating_counts >= R_min) & (user_rating_counts <= R_max)].index
        filtered_df = self.df[self.df['User'].isin(filtered_users)]
        filtered_items = filtered_df['Item'].value_counts().index
        final_df = filtered_df[filtered_df['Item'].isin(filtered_items)]
        return final_df, filtered_users.size, filtered_items.size

    def plot_histograms(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        # Frequency histogram of ratings per user
        user_ratings_count = self.df['User'].value_counts()
        axes[0].hist(user_ratings_count, bins=30, color='blue', alpha=0.7)
        axes[0].set_title('Frequency of Ratings per User')
        axes[0].set_xlabel('Number of Ratings')
        axes[0].set_ylabel('Number of Users')
        axes[0].set_xlim([user_ratings_count.min(), user_ratings_count.max()])

        # Time span histogram of ratings per user
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        user_date_range = self.df.groupby('User')['Date'].agg([min, max])
        user_date_range['range'] = user_date_range['max'] - user_date_range['min']
        user_date_range['range'] = pd.to_timedelta(user_date_range['range']).dt.total_seconds() / (24 * 60 * 60)
        user_date_range['range'] = pd.to_datetime(user_date_range['range'], unit='D')

        axes[1].hist(user_date_range['range'], bins=30, color='green', alpha=0.7)
        axes[1].set_title('Time Span of Ratings per User')
        axes[1].set_xlabel('Date Range')
        axes[1].set_ylabel('Number of Users')
        axes[1].set_xlim([user_date_range['range'].min(), user_date_range['range'].max()])

        plt.tight_layout()
        plt.show()

    def create_preference_vectors(self):
        user_item_matrix = self.df.pivot_table(index='User', columns='Item', values='Rating', aggfunc='max')
        user_preference_vectors = user_item_matrix.fillna(0)
        return user_preference_vectors


    def create_preference_vectors_given_bounds(self, top_users1, top_items1):
        top_items = self.df['Item'].value_counts().nlargest(top_items1).index
        top_users = self.df['User'].value_counts().nlargest(top_users1).index
        # Filter data to include only the top most frequently rated items
        filtered_df = self.df[self.df['Item'].isin(top_items)]

        # Filter data to include only the top users who have rated the most items
        filtered_df = filtered_df[filtered_df['User'].isin(top_users)]

        # Now create the pivot table
        user_item_matrix = filtered_df.pivot_table(index='User', columns='Item', values='Rating', aggfunc='max')

        # Fill NaN values with 0, which represent unrated items
        user_preference_vectors = user_item_matrix.fillna(0)

        return user_preference_vectors

# Usage
# processor = DataProcessor('../Dataset.csv')

# Exaple preference vectors with bounds to lower the error margin
# top_items = df['Item'].value_counts().nlargest(1000).index
# top_users = df['User'].value_counts().nlargest(1000).index
#
# preference_vectors = processor.create_preference_vectors(top_users, top_items)
# print(preference_vectors.head())