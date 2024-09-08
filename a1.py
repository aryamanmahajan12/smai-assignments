import numpy as np
import pandas as pd
from collections import Counter

def calculate_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    
    return accuracy

def convert_columns_to_numeric(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def min_max_normalize(X):
    X = X.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    denominator = col_max - col_min
    denominator[denominator == 0] = 1 
    X_normalized = (X - col_min) / denominator
    return X_normalized

def manhattan_distance(X1, X2):
    return np.abs(X1[:, np.newaxis] - X2).sum(axis=2)

def euclidean_distance(X1, X2):
    return np.sqrt(np.sum(np.square(X1[:, np.newaxis] - X2), axis=2))

def cosine_distance(X1, X2):
    X1_normalized = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    X2_normalized = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    cosine_similarity = np.dot(X1_normalized, X2_normalized.T)
    return 1 - cosine_similarity

def most_common_genre(nearest_genres_row):
    genre_list = nearest_genres_row.tolist()
    most_common = Counter(genre_list).most_common(1)
    return most_common[0][0]


class KNN:

    def __init__(self, k, distance_metric='cosine'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        if self.distance_metric == 'cosine':
            distances = cosine_distance(X, self.X_train)
        elif self.distance_metric == 'manhattan':
            distances = manhattan_distance(X, self.X_train)
        elif self.distance_metric == 'euclidean':
            distances = euclidean_distance(X, self.X_train)
        else:
            raise ValueError("Unsupported distance metric")
        
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        nearest_genres = self.Y_train[k_indices]
        
        predictions = np.apply_along_axis(most_common_genre, 1, nearest_genres)
        
        return predictions
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree, reg_type='none', lambda_val=0):
        self.degree = degree
        self.reg_type = reg_type
        self.lambda_val = lambda_val
        self.coeffs = None

    def polynomial_features(self, X):
        return np.column_stack([X**i for i in range(1, self.degree+1)])

    def fit(self, X, y):
        X_poly = self.polynomial_features(X)
        n_features = X_poly.shape[1]

        if self.reg_type == 'none':
            self.coeffs = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        elif self.reg_type == 'l2':
            self.coeffs = np.linalg.inv(X_poly.T @ X_poly + self.lambda_val * np.eye(n_features)) @ X_poly.T @ y
        elif self.reg_type == 'l1':
            self.coeffs = np.zeros(n_features)
            max_iter = 1000
            tol = 1e-4
            for _ in range(max_iter):
                coeffs_old = self.coeffs.copy()
                for j in range(n_features):
                    y_pred = X_poly @ self.coeffs
                    r = y - y_pred + self.coeffs[j] * X_poly[:, j]
                    z_j = X_poly[:, j] @ r
                    if z_j > self.lambda_val:
                        self.coeffs[j] = (z_j - self.lambda_val) / (X_poly[:, j] @ X_poly[:, j])
                    elif z_j < -self.lambda_val:
                        self.coeffs[j] = (z_j + self.lambda_val) / (X_poly[:, j] @ X_poly[:, j])
                    else:
                        self.coeffs[j] = 0
                if np.sum(np.abs(self.coeffs - coeffs_old)) < tol:
                    break

    def predict(self, X):
        X_poly = self.polynomial_features(X)
        return X_poly @ self.coeffs

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def variance(y):
    return np.var(y)

def standard_deviation(y):
    return np.std(y)

train_data = pd.read_csv('reg_train.csv')
val_data = pd.read_csv('reg_val.csv')
test_data = pd.read_csv('reg_test.csv')

X_train, y_train = train_data['x'].values, train_data['y'].values
X_val, y_val = val_data['x'].values, val_data['y'].values
X_test, y_test = test_data['x'].values, test_data['y'].values

max_degree = 20
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
reg_types = ['none', 'l1', 'l2']

results = []

for reg_type in reg_types:
    for lambda_val in lambda_values:
        for degree in range(1, max_degree + 1):
            model = PolynomialRegression(degree, reg_type, lambda_val)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_val = mean_squared_error(y_val, y_val_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            
            var_train = variance(y_train_pred)
            var_val = variance(y_val_pred)
            var_test = variance(y_test_pred)
            
            std_train = standard_deviation(y_train_pred)
            std_val = standard_deviation(y_val_pred)
            std_test = standard_deviation(y_test_pred)
            
            results.append({
                'reg_type': reg_type,
                'lambda': lambda_val,
                'degree': degree,
                'mse_train': mse_train,
                'mse_val': mse_val,
                'mse_test': mse_test,
                'var_train': var_train,
                'var_val': var_val,
                'var_test': var_test,
                'std_train': std_train,
                'std_val': std_val,
                'std_test': std_test,
                'model': model
            })

plt.figure(figsize=(20, 15))
X_plot = np.linspace(min(X_train.min(), X_val.min(), X_test.min()),
                     max(X_train.max(), X_val.max(), X_test.max()),
                     1000).reshape(-1, 1)

for i, reg_type in enumerate(reg_types):
    plt.subplot(1, 3, i+1)
    plt.scatter(X_train, y_train, color='blue', label='Training data', alpha=0.5)
    plt.scatter(X_val, y_val, color='green', label='Validation data', alpha=0.5)
    plt.scatter(X_test, y_test, color='red', label='Test data', alpha=0.5)

    if reg_type == 'none':
        degrees_to_plot = [1, 5, 20]  
        for degree in degrees_to_plot:
            model = PolynomialRegression(degree, reg_type, 0)
            model.fit(X_train, y_train)
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    else:
        for lambda_val in [0.0001, 0.1, 10]:
            best_model = min([r for r in results if r['reg_type'] == reg_type and r['lambda'] == lambda_val], 
                             key=lambda x: x['mse_val'])
            y_plot = best_model['model'].predict(X_plot)
            plt.plot(X_plot, y_plot, label=f'λ={lambda_val}, degree={best_model["degree"]}')

    plt.legend()
    plt.title(f'{reg_type.upper()} Regularization')
    plt.xlabel('X')
    plt.ylabel('Y')

plt.tight_layout()
plt.show()

for reg_type in reg_types:
    print(f"\nBest models for {reg_type.upper()} regularization:")
    for lambda_val in lambda_values:
        best_model = min([r for r in results if r['reg_type'] == reg_type and r['lambda'] == lambda_val], 
                         key=lambda x: x['mse_val'])
        print(f"\nλ = {lambda_val}")
        print(f"Best degree: {best_model['degree']}")
        print(f"MSE (Train/Val/Test): {best_model['mse_train']:.4f} / {best_model['mse_val']:.4f} / {best_model['mse_test']:.4f}")
        print(f"Variance (Train/Val/Test): {best_model['var_train']:.4f} / {best_model['var_val']:.4f} / {best_model['var_test']:.4f}")
        print(f"Std Dev (Train/Val/Test): {best_model['std_train']:.4f} / {best_model['std_val']:.4f} / {best_model['std_test']:.4f}")



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