
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Use predict_proba to get probability estimates
probabilities = model.predict_proba(X_test)

# Print the probabilities
print("Probabilities for the first test instance:", probabilities[0])

# Optional: Print the first test instance and its true label for reference
print("First test instance:", X_test[0])
print("True label for the first test instance:", y_test[0])