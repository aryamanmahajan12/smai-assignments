import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        self.slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, x):
        return self.slope * x + self.intercept

    def plot(self, x, y):
        y_fit = self.predict(x)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='b', label='Training Data')
        plt.plot(x, y_fit, color='r', label='Best-Fit Line')
        plt.xlabel('X Axis Label')  
        plt.ylabel('Y Axis Label') 
        plt.title('X vs Y with Best-Fit Line')
        plt.legend()
        plt.show()

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    variance = np.var(y_true)
    std_dev = np.std(y_true)
    return mse, variance, std_dev

train_file = 'train.csv'  
test_file = 'val.csv'  

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


x_train = train_data['x'].values
y_train = train_data['y'].values
x_test = test_data['x'].values
y_test = test_data['y'].values

model = LinearRegression()

# Fit the model to the training data
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

mse_train, var_train, std_train = calculate_metrics(y_train, y_train_pred)

mse_test, var_test, std_test = calculate_metrics(y_test, y_test_pred)

print(f"Training Data - MSE: {mse_train:.4f}, Variance: {var_train:.4f}, Std Dev: {std_train:.4f}")
print(f"Testing Data - MSE: {mse_test:.4f}, Variance: {var_test:.4f}, Std Dev: {std_test:.4f}")

model.plot(x_train, y_train)
