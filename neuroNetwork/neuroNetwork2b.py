from dataClass import DataProcessor
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load the CSV file

data = DataProcessor('../Dataset.csv')

user_preference_vectors = data.create_preference_vectors_given_bounds(2000, 2000)

# Show the transformed DataFrame
print(user_preference_vectors.head())

# Identify k-nearest neighbors within the same cluster for each user
def identify_neighbors(user_preference_vectors, k):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k, n_jobs=-1)
    model_knn.fit(user_preference_vectors)

    distances, indices = model_knn.kneighbors(user_preference_vectors)

    return indices

# Construct feature vectors for each user
def construct_feature_vectors(user_preference_vectors, indices):
    feature_vectors = np.array([user_preference_vectors.iloc[indices[i]].values.flatten() for i in range(len(indices))])

    return feature_vectors

# Define and train a neural network for each cluster
def train_neural_network(feature_vectors, target_vectors):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, target_vectors, test_size=0.2, random_state=42)

    # Define the neural network
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

    return model

# Predict ratings
def predict_ratings(model, feature_vectors):
    predicted_ratings = model.predict(feature_vectors)

    return predicted_ratings

# Usage
k = 5  # Number of neighbors
indices = identify_neighbors(user_preference_vectors, k)
feature_vectors = construct_feature_vectors(user_preference_vectors, indices)
model = train_neural_network(feature_vectors, user_preference_vectors.values)
predicted_ratings = predict_ratings(model, feature_vectors)
print(predicted_ratings)

# Calculate the mean squared error (MSE) between the actual and predicted ratings

from sklearn.metrics import mean_squared_error

# Flatten the arrays for compatibility with sklearn's mean_squared_error
flat_actual_ratings = user_preference_vectors.values.flatten()
flat_predicted_ratings = predicted_ratings.flatten()

# Calculate the mean squared error
mse = mean_squared_error(flat_actual_ratings, flat_predicted_ratings)

print(f"Mean Squared Error: {mse}")