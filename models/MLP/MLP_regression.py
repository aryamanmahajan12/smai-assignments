import numpy as np
import pandas as pd
class MLPRegression:
    def __init__(self, hidden_layers, activations, learning_rate=0.01, optimizer='sgd', batch_size=32, epochs=100, early_stopping_patience=10):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activations = activations
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weights = []
        self.biases = []

    def _initialize_parameters(self, input_size):
        layer_sizes = [input_size] + self.hidden_layers + [1]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i])))

    def _activation_function(self, x, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'linear':
            return x

    def _activation_derivative(self, x, activation):
        if activation == 'sigmoid':
            return x * (1 - x)
        elif activation == 'tanh':
            return 1 - np.power(x, 2)
        elif activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif activation == 'linear':
            return np.ones_like(x)

    def _forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self._activation_function(z, self.activations[i])
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def _backward_propagation(self, X, y, y_pred):
        m = X.shape[0]
        dZ = y_pred - y
        gradients = []
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.layer_outputs[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients.append((dW, db))
            if i > 0:
                dZ = np.dot(dZ, self.weights[i].T) * self._activation_derivative(self.layer_outputs[i], self.activations[i-1])
        return gradients[::-1]

    def _update_parameters(self, gradients):
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients[i][0]
                self.biases[i] -= self.learning_rate * gradients[i][1]
        elif self.optimizer == 'batch':
            # Implement batch gradient descent
            pass
        elif self.optimizer == 'mini_batch':
            # Implement mini-batch gradient descent
            pass

    def fit(self, X, y):
        self._initialize_parameters(X.shape[1])
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                indices = np.random.permutation(X.shape[0])
                X = X[indices]
                y = y[indices]
            
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                
                y_pred = self._forward_propagation(X_batch)
                gradients = self._backward_propagation(X_batch, y_batch, y_pred)
                self._update_parameters(gradients)
            
            # Compute loss for early stopping
            y_pred = self._forward_propagation(X)
            loss = np.mean((y_pred - y) ** 2)
            if(epoch%10 ==0):
                print(f"loss: {loss}")
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return self

    def predict(self, X):
        X = np.array(X)
        return self._forward_propagation(X)
