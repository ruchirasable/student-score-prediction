# ------------------------------
# Predicting Student Exam Scores using Linear Regression
# By Ruchira Sable
# ------------------------------

# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("\nStep 1: Loading the dataset...\n")

# 2. Load the dataset
data = pd.read_csv("student_scores.csv")
print("Here are the first 5 rows of the dataset:")
print(data.head())

# 3. Select features (inputs) and target (output)
print("\nStep 2: Selecting features and target...")
X = data[['hours_studied', 'sleep_hours', 'attendance']]
y = data['score']
print("Features and target selected successfully.")

# 4. Split the dataset into training and testing sets
print("\nStep 3: Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression model
print("Step 4: Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# 6. Make predictions
print("\nStep 5: Making predictions on test data...")
pred = model.predict(X_test)
print("Predictions generated!")

# 7. Evaluate the model
print("\nStep 6: Model Evaluation ----")
print("Mean Squared Error:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# 8. Visualize results
print("\nStep 7: Visualizing Actual vs Predicted Scores...")
plt.scatter(y_test, pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted Student Exam Scores")
plt.show()
