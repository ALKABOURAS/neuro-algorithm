import numpy as np
import pandas as pd

# Load the numpy array
# Load the numpy array
data = np.load('Dataset.npy')

# Split the data into separate columns
data = [item.split(',') for item in data]

# Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(data, columns=['User', 'Item', 'Rating', 'Date'])

# Print the first 5 rows of the DataFrame
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('Dataset.csv', index=False)