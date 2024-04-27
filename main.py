from generate_data import generate_trader_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import numpy as np


# Load and preprocess the data
data = generate_trader_data()
data['transaction_made'] = 1 # All transactions in your dataset are positive
# Convert the 'date' column to numerical format
data['date'] = pd.to_numeric(data['date'])

# Continue with the rest of your code
# Prepare your features and target variable
X = data[['price', 'amount', 'date', 'marketcap']]
y = data['transaction_made']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Assuming 'knn' is your trained KNN classifier
# And 'sample_transaction' is a numpy array or a pandas DataFrame containing the features of the sample transaction

# Get the current date in the format YYYYMMDD
current_date = datetime.now().strftime('%Y%m%d')

# Create a sample transaction with values for price, amount, date, and marketcap
sample_transaction = np.array([[10, 0.5, int(current_date), 5000000]])

# Ensure it's a 2D array
sample_transaction = np.array([sample_transaction])

# Reshape the array to make it 2D
sample_transaction = np.reshape(sample_transaction, (1, -1))

# Use predict_proba to get the probabilities
probabilities = knn.predict_proba(sample_transaction)

# The output is an array of probabilities for each class. Since it's a binary classification,
# the first column represents the probability of the transaction not being made, and the second column
# represents the probability of the transaction being made.
# To get the percentage prediction for the transaction being made, you can take the first column.

percentage_prediction = probabilities[0][0] * 100

print(f"The model predicts a {percentage_prediction:.2f}% chance of the transaction being made.")