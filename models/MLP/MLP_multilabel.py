import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFunction.sigmoid(x) * (1 - ActivationFunction.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid'):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = getattr(ActivationFunction, activation)
        self.activation_derivative = getattr(ActivationFunction, f"{activation}_derivative")

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.activation(self.z)

    def backward(self, delta):
        d_z = delta * self.activation_derivative(self.z)
        d_weights = np.dot(self.inputs.T, d_z)
        d_biases = np.sum(d_z, axis=0, keepdims=True)
        d_inputs = np.dot(d_z, self.weights.T)
        return d_weights, d_biases, d_inputs

class MultiLabelMLP:
    def __init__(self, input_size, output_size, hidden_layers=[64, 64], activations=['sigmoid', 'sigmoid', 'sigmoid'], learning_rate=0.01, epochs=100, batch_size=32, early_stopping_patience=10):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y_true, y_pred):
        delta = binary_cross_entropy_derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            d_weights, d_biases, delta = layer.backward(delta)
            layer.weights -= self.learning_rate * d_weights
            layer.biases -= self.learning_rate * d_biases

    def fit(self, X_train, y_train, X_val=None, y_val=None, optimizer='sgd'):

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(self.epochs):
            if optimizer == 'sgd':
                self._fit_epoch_sgd(X_train, y_train)
            elif optimizer == 'batch':
                self._fit_epoch_batch(X_train, y_train)
            elif optimizer == 'mini_batch':
                self._fit_epoch_mini_batch(X_train, y_train)
            else:
                raise ValueError("Invalid optimizer. Choose 'sgd', 'batch', or 'mini_batch'.")

            val_loss = binary_cross_entropy_loss(y_val, self.forward(X_val))
            
            if (epoch + 1) % 10 == 0:
                train_loss = binary_cross_entropy_loss(y_train, self.forward(X_train))
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [layer.weights.copy() for layer in self.layers]
                best_biases = [layer.biases.copy() for layer in self.layers]
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best weights and biases
        if best_weights is not None and best_biases is not None:
            for i, layer in enumerate(self.layers):
                layer.weights = best_weights[i]
                layer.biases = best_biases[i]

    def _fit_epoch_sgd(self, X_train, y_train):
        for i in range(len(X_train)):
            X = X_train[i:i+1]
            y = y_train[i:i+1]
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)

    def _fit_epoch_batch(self, X_train, y_train):
        y_pred = self.forward(X_train)
        self.backward(X_train, y_train, y_pred)

    def _fit_epoch_mini_batch(self, X_train, y_train):
        for i in range(0, len(X_train), self.batch_size):
            X_batch = X_train[i:i+self.batch_size]
            y_batch = y_train[i:i+self.batch_size]
            y_pred = self.forward(X_batch)
            self.backward(X_batch, y_batch, y_pred)

    def predict(self, X, threshold=0.5):
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)

    def evaluate(self, X_test, y_test, threshold=0.5):
        y_pred = self.predict(X_test, threshold)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='samples')
        recall = recall_score(y_test, y_pred, average='samples')
        f1 = f1_score(y_test, y_pred, average='samples')
        hamming = hamming_loss(y_test, y_pred)
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Hamming Loss: {hamming:.4f}')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hamming_loss': hamming
        }