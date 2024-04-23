# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the CSV file
# df = pd.read_csv('../Dataset.csv')
#
# # Plotting histograms
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
#
# # Frequency histogram of ratings per user
# user_ratings_count = df['User'].value_counts()
# axes[0].hist(user_ratings_count, bins=30, color='blue', alpha=0.7)
# axes[0].set_title('Frequency of Ratings per User')
# axes[0].set_xlabel('Number of Ratings')
# axes[0].set_ylabel('Number of Users')
# # Set x-axis limits to match min and max of the data
# axes[0].set_xlim([user_ratings_count.min(), user_ratings_count.max()])
#
# # Time span histogram of ratings per user
# df['Date'] = pd.to_datetime(df['Date'])
# user_date_range = df.groupby('User')['Date'].agg([min, max])
# user_date_range['range'] = (user_date_range['max'] - user_date_range['min']).dt.days
# axes[1].hist(user_date_range['range'], bins=30, color='green', alpha=0.7)
# axes[1].set_title('Time Span of Ratings per User')
# axes[1].set_xlabel('Days')
# axes[1].set_ylabel('Number of Users')
# # Set x-axis limits to match min and max of the data
# axes[1].set_xlim([user_date_range['range'].min(), user_date_range['range'].max()])
#
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('../Dataset.csv')

# Plotting histograms
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Frequency histogram of ratings per user
user_ratings_count = df['User'].value_counts()
axes[0].hist(user_ratings_count, bins=30, color='blue', alpha=0.7)
axes[0].set_title('Frequency of Ratings per User')
axes[0].set_xlabel('Number of Ratings')
axes[0].set_ylabel('Number of Users')
# Set x-axis limits to match min and max of the data
axes[0].set_xlim([user_ratings_count.min(), user_ratings_count.max()])

# Time span histogram of ratings per user
df['Date'] = pd.to_datetime(df['Date'])
user_date_range = df.groupby('User')['Date'].agg([min, max])
user_date_range['range'] = user_date_range['max'] - user_date_range['min']

# Convert the 'range' column to datetime format
user_date_range['range'] = pd.to_timedelta(user_date_range['range']).dt.total_seconds() / (24 * 60 * 60)
user_date_range['range'] = pd.to_datetime(user_date_range['range'], unit='D')

# Plot a histogram of the range of dates
axes[1].hist(user_date_range['range'], bins=30, color='green', alpha=0.7)
axes[1].set_title('Time Span of Ratings per User')
axes[1].set_xlabel('Date Range')
axes[1].set_ylabel('Number of Users')

# Set x-axis limits to match min and max of the data
axes[1].set_xlim([user_date_range['range'].min(), user_date_range['range'].max()])

plt.tight_layout()
plt.show()
