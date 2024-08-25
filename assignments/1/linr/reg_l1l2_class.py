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