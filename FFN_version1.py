import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Generate a simple dataset (e.g., XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([[0], [1], [1], [0]])              # Outputs (labels)

# Activation function and its derivative (Sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Loss function (Mean Squared Error)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Initialize weights and biases
input_size = 2    # Number of input neurons
hidden_size = 8   # Number of hidden neurons
output_size = 1   # Number of output neurons

W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = np.zeros((1, hidden_size))                      # Biases for hidden layer
W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size) # Weights for hidden -> output
b2 = np.zeros((1, output_size))                      # Biases for output layer

# Hyperparameters
learning_rate = 0.01
epochs = 10000

# Training loop
# Training loop
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(X, W1) + b1           # Linear combination (input -> hidden)
    A1 = sigmoid(Z1)                  # Activation (hidden layer)
    Z2 = np.dot(A1, W2) + b2          # Linear combination (hidden -> output)
    A2 = sigmoid(Z2)                  # Activation (output layer)

    # Calculate loss
    loss = mse_loss(y, A2)

    # Backward propagation
    dA2 = 2 * (A2 - y) / y.size       # Derivative of loss w.r.t output
    dZ2 = dA2 * sigmoid_derivative(Z2)  # Backprop through output activation
    dW2 = np.dot(A1.T, dZ2)           # Gradient for W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # Gradient for b2

    dA1 = np.dot(dZ2, W2.T)           # Backprop into hidden layer
    dZ1 = dA1 * sigmoid_derivative(Z1)  # Backprop through hidden activation
    dW1 = np.dot(X.T, dZ1)            # Gradient for W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # Gradient for b1

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the trained model
print("\nTrained Model Predictions:")
for i in range(len(X)):
    Z1 = np.dot(X[i], W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    print(f"Input: {X[i]}, Predicted: {A2[0, 0]:.4f}, Actual: {y[i][0]}")


# Testing the trained model
print("\nTrained Model Predictions:")
for i in range(len(X)):
    Z1 = np.dot(X[i], W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
   
    print(f"Input: {X[i]}, Predicted: {A2[0, 0]:.4f}, Actual: {y[i][0]}")


