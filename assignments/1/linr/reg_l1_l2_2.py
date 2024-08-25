import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv('reg_train.csv')
val_data = pd.read_csv('reg_val.csv')
test_data = pd.read_csv('reg_test.csv')

# Assuming 'x' is the input feature and 'y' is the target variable
X_train, y_train = train_data['x'].values, train_data['y'].values
X_val, y_val = val_data['x'].values, val_data['y'].values
X_test, y_test = test_data['x'].values, test_data['y'].values

def polynomial_features(X, degree):
    return np.column_stack([X**i for i in range(1, degree+1)])

def fit_polynomial_regression(X, y, lambda_val, reg_type='l2'):
    n_features = X.shape[1]
    if reg_type == 'l2':
        # Ridge regression
        return np.linalg.inv(X.T @ X + lambda_val * np.eye(n_features)) @ X.T @ y
    elif reg_type == 'l1':
        # Lasso regression (using coordinate descent)
        max_iter = 1000
        tol = 1e-4
        coeffs = np.zeros(n_features)
        for _ in range(max_iter):
            coeffs_old = coeffs.copy()
            for j in range(n_features):
                y_pred = X @ coeffs
                r = y - y_pred + coeffs[j] * X[:, j]
                z_j = X[:, j] @ r
                if z_j > lambda_val:
                    coeffs[j] = (z_j - lambda_val) / (X[:, j] @ X[:, j])
                elif z_j < -lambda_val:
                    coeffs[j] = (z_j + lambda_val) / (X[:, j] @ X[:, j])
                else:
                    coeffs[j] = 0
            if np.sum(np.abs(coeffs - coeffs_old)) < tol:
                break
        return coeffs

def predict(X, coeffs):
    return X @ coeffs

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def variance(y):
    return np.var(y)

def standard_deviation(y):
    return np.std(y)

max_degree = 20
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
reg_types = ['l1', 'l2']

results = []

for reg_type in reg_types:
    for lambda_val in lambda_values:
        for degree in range(1, max_degree + 1):
            # Generate polynomial features
            X_train_poly = polynomial_features(X_train, degree)
            X_val_poly = polynomial_features(X_val, degree)
            X_test_poly = polynomial_features(X_test, degree)
            
            # Fit the model
            coeffs = fit_polynomial_regression(X_train_poly, y_train, lambda_val, reg_type)
            
            # Make predictions
            y_train_pred = predict(X_train_poly, coeffs)
            y_val_pred = predict(X_val_poly, coeffs)
            y_test_pred = predict(X_test_poly, coeffs)
            
            # Calculate metrics
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
                'coeffs': coeffs
            })

# Plotting
plt.figure(figsize=(20, 15))
X_plot = np.linspace(min(X_train.min(), X_val.min(), X_test.min()),
                     max(X_train.max(), X_val.max(), X_test.max()),
                     1000).reshape(-1, 1)

for i, reg_type in enumerate(reg_types):
    plt.subplot(1, 2, i+1)
    plt.scatter(X_train, y_train, color='blue', label='Training data', alpha=0.5)
    plt.scatter(X_val, y_val, color='green', label='Validation data', alpha=0.5)
    plt.scatter(X_test, y_test, color='red', label='Test data', alpha=0.5)

    for lambda_val in [0.0001, 0.1, 10]:
        best_model = min([r for r in results if r['reg_type'] == reg_type and r['lambda'] == lambda_val], 
                         key=lambda x: x['mse_val'])
        X_plot_poly = polynomial_features(X_plot, best_model['degree'])
        y_plot = predict(X_plot_poly, best_model['coeffs'])
        plt.plot(X_plot, y_plot, label=f'{reg_type}, λ={lambda_val}, degree={best_model["degree"]}')

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