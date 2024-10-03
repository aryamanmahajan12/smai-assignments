import numpy as np

class MLPClassifier:

    def __init__(self, hidden_layers=[64, 32], learning_rate=0.01, activation='relu', 
                 optimizer='sgd', batch_size=32, epochs=100):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.label_map = None
        self.inverse_label_map = None
        
    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01)   
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def _activate(self, X, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif activation == 'tanh':
            return np.tanh(X)
        elif activation == 'relu':
            return np.maximum(0, X)
        elif activation == 'linear':
            return X
        elif activation == 'softmax':
            exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function")
    
    def _activate_derivative(self, X, activation):
        if activation == 'sigmoid':
            return X * (1 - X)
        elif activation == 'tanh':
            return 1 - np.power(X, 2)
        elif activation == 'relu':
            return (X > 0).astype(float)
        elif activation == 'linear':
            return np.ones_like(X)
        elif activation == 'softmax':
            return X * (1 - X)  # This is not used directly for softmax
        else:
            raise ValueError("Unsupported activation function")
    


    
    def forward_propagation(self, X):

        self.layer_outputs = [X]


        for i in range(len(self.weights)):

            if i == len(self.weights) - 1:  # Output layer
                Z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
                A = self._activate(Z, 'softmax')  

            else:
                Z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
                A = self._activate(Z, self.activation)

            self.layer_outputs.append(A)

        return self.layer_outputs[-1]
    


    def backpropagation(self, X, y, y_pred):
        
        m = X.shape[0]
        self.gradients = []
        
        # Output layer
        dZ = y_pred - y
        dW = (1/m) * np.dot(self.layer_outputs[-2].T, dZ)
        db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        self.gradients.append((dW, db))
        
        # Hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self._activate_derivative(self.layer_outputs[i+1], self.activation)
            dW = (1/m) * np.dot(self.layer_outputs[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            self.gradients.append((dW, db))
        
        self.gradients = self.gradients[::-1]

    def update_parameters(self):
        for i in range(len(self.weights)):
            if self.optimizer in ['batch', 'sgd', 'mini-batch']:
                self.weights[i] -= self.learning_rate * self.gradients[i][0]
                self.biases[i] -= self.learning_rate * self.gradients[i][1]
            else:
                raise ValueError("Unsupported optimizer")
    
    def fit(self, X, y):
        # Create label mapping
        unique_labels = np.unique(y)
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.inverse_label_map = {i: label for label, i in self.label_map.items()}
        
        input_size = X.shape[1]
        output_size = len(unique_labels)
        self._initialize_parameters(input_size, output_size)
        
        y_encoded = self._encode_labels(y)
        y_onehot = self._onehot_encode(y_encoded)
        
        best_loss = float('inf')
        patience = 10
        counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer == 'batch':
                y_pred = self.forward_propagation(X)
                self.backpropagation(X, y_onehot, y_pred)
                self.update_parameters()
            elif self.optimizer in ['sgd', 'mini-batch']:
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                for start_idx in range(0, X.shape[0], self.batch_size):
                    batch_indices = indices[start_idx:start_idx+self.batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y_onehot[batch_indices]
                    
                    y_pred_batch = self.forward_propagation(X_batch)
                    self.backpropagation(X_batch, y_batch, y_pred_batch)
                    self.update_parameters()
            
            # Early stopping
            current_loss = self._compute_loss(X, y_onehot)
            if current_loss < best_loss:
                best_loss = current_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss}")
    
    def predict(self, X):
        probabilities = self.forward_propagation(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return np.array([self.inverse_label_map[i] for i in predicted_indices])
    
    def _encode_labels(self, y):
        return np.array([self.label_map[label] for label in y])
    
    def _onehot_encode(self, y):
        if len(y.shape) == 1:
            n_classes = len(np.unique(y))
            return np.eye(n_classes)[y]
        return y
    
    def _compute_loss(self, X, y):
        y_pred = self.forward_propagation(X)
        return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))