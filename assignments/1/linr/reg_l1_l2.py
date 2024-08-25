import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('reg_train.csv')
val_data = pd.read_csv('reg_val.csv')
test_data = pd.read_csv('reg_test.csv')

X_train, y_train = train_data['x'].values, train_data['y'].values
X_val, y_val = val_data['x'].values, val_data['y'].values
X_test, y_test = test_data['x'].values, test_data['y'].values

def polynomial_features(X, degree):
    return np.column_stack([X**i for i in range(1, degree+1)])

def fit_polynomial_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def predict(X, coeffs):
    return X @ coeffs

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def variance(y):
    return np.var(y)

def standard_deviation(y):
    return np.std(y)

max_degree = 20
results = []

for degree in range(1, max_degree + 1):
    X_train_poly = polynomial_features(X_train, degree)
    X_val_poly = polynomial_features(X_val, degree)
    X_test_poly = polynomial_features(X_test, degree)
    
    coeffs = fit_polynomial_regression(X_train_poly, y_train)
    
    y_train_pred = predict(X_train_poly, coeffs)
    y_val_pred = predict(X_val_poly, coeffs)
    y_test_pred = predict(X_test_poly, coeffs)
    
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
plt.figure(figsize=(15, 10))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_val, y_val, color='green', label='Validation data')
plt.scatter(X_test, y_test, color='red', label='Test data')

X_plot = np.linspace(min(X_train.min(), X_val.min(), X_test.min()),
                     max(X_train.max(), X_val.max(), X_test.max()),
                     1000).reshape(-1, 1)

for degree in [1, 5, 10, 20]:
    X_plot_poly = polynomial_features(X_plot, degree)
    y_plot = predict(X_plot_poly, results[degree-1]['coeffs'])
    plt.plot(X_plot, y_plot, label=f'Degree {degree}')

plt.legend()
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Print results
for result in results:
    print(f"Degree: {result['degree']}")
    print(f"MSE (Train/Val/Test): {result['mse_train']:.4f} / {result['mse_val']:.4f} / {result['mse_test']:.4f}")
    print(f"Variance (Train/Val/Test): {result['var_train']:.4f} / {result['var_val']:.4f} / {result['var_test']:.4f}")
    print(f"Std Dev (Train/Val/Test): {result['std_train']:.4f} / {result['std_val']:.4f} / {result['std_test']:.4f}")
    print()