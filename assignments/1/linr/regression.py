import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = None

    def fit(self, x, y):
        # Create the design matrix for polynomial regression
        X = np.vander(x, self.degree + 1)
        # Calculate the coefficients using the normal equation
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, x):
        # Create the design matrix for prediction
        X = np.vander(x, self.degree + 1)
        # Predict the values using the coefficients
        return X @ self.coefficients

    def plot(self, x, y):
        # Predict the y values using the fitted model
        x_curve = np.linspace(min(x), max(x), 500)
        y_curve = self.predict(x_curve)
        
        # Plot the original data points as a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='b', label='Data Points')
        # Plot the fitted polynomial curve as a smooth line
        plt.plot(x_curve, y_curve, color='r', label=f'Polynomial Fit (degree={self.degree})')
        plt.xlabel('X Axis Label')  # Replace with your x-axis label
        plt.ylabel('Y Axis Label')  # Replace with your y-axis label
        plt.title(f'Polynomial Regression (degree={self.degree})')
        plt.legend()
        plt.show()

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    variance = np.var(y_true)
    std_dev = np.std(y_true)
    return mse, variance, std_dev

# Load training and testing data
train_file = 'train.csv'  # Replace with your training CSV file path
test_file = 'val.csv'  # Replace with your testing CSV file path

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Extract x and y columns
x_train = train_data['x'].values
y_train = train_data['y'].values
x_test = test_data['x'].values
y_test = test_data['y'].values

# List of polynomial degrees to test
degrees = [1,2, 3, 4, 5, 6]  # Modify this list as needed

# Initialize lists to store results
results = []

# Iterate over each polynomial degree
for k in degrees:
    # Create an instance of PolynomialRegression with degree k
    model = PolynomialRegression(degree=k)

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Predict on both training and testing data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate metrics for training data
    mse_train, var_train, std_train = calculate_metrics(y_train, y_train_pred)

    # Calculate metrics for testing data
    mse_test, var_test, std_test = calculate_metrics(y_test, y_test_pred)

    # Store the results
    results.append({
        'degree': k,
        'mse_train': mse_train,
        'variance_train': var_train,
        'std_dev_train': std_train,
        'mse_test': mse_test,
        'variance_test': var_test,
        'std_dev_test': std_test
    })

# Print the results
for result in results:
    print(f"Degree {result['degree']}:")
    print(f"  Training Data - MSE: {result['mse_train']:.4f}, Variance: {result['variance_train']:.4f}, Std Dev: {result['std_dev_train']:.4f}")
    print(f"  Testing Data - MSE: {result['mse_test']:.4f}, Variance: {result['variance_test']:.4f}, Std Dev: {result['std_dev_test']:.4f}")
    print()

# Find the degree that minimizes the MSE on the test set
best_degree = min(results, key=lambda x: x['mse_test'])['degree']
print(f"The degree that minimizes the error on the test set is: {best_degree}")


# Find the degree that minimizes the MSE on the test set
best_degree = min(results, key=lambda x: x['mse_test'])['degree']
print(f"The degree that minimizes the error on the test set is: {best_degree}")

# Train the best model again
best_model = PolynomialRegression(degree=best_degree)
best_model.fit(x_train, y_train)

# Save the coefficients of the best model
with open('best_model_coefficients.pkl', 'wb') as f:
    pickle.dump(best_model.coefficients, f)

print(f"The coefficients of the best model (degree {best_degree}) have been saved to 'best_model_coefficients.pkl'")

# Optional: Demonstrate how to load and use the saved coefficients
with open('best_model_coefficients.pkl', 'rb') as f:
    loaded_coefficients = pickle.load(f)

print("Loaded coefficients:", loaded_coefficients)

# Create a new model with the loaded coefficients
loaded_model = PolynomialRegression(degree=best_degree)
loaded_model.coefficients = loaded_coefficients

# Test the loaded model
y_test_pred_loaded = loaded_model.predict(x_test)
mse_test_loaded = np.mean((y_test - y_test_pred_loaded) ** 2)
print(f"MSE of the loaded model on test data: {mse_test_loaded:.4f}")