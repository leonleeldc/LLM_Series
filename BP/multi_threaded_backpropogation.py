import threading
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.lock = threading.Lock()  # Lock for thread synchronization

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.a1_error = self.output_delta.dot(self.W2.T)
        self.a1_delta = self.a1_error * self.sigmoid_derivative(self.a1)

        self.W1_grad = X.T.dot(self.a1_delta)
        self.W2_grad = self.a1.T.dot(self.output_delta)

        # Use lock to synchronize gradient updates
        with self.lock:
            self.W1 += self.W1_grad
            self.W2 += self.W2_grad

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

    def train_multithreaded(self, X, y, epochs, batch_size, num_threads):
        def worker(thread_id, X, y):
            for epoch in range(epochs):
                for i in range(thread_id * batch_size, len(X), num_threads * batch_size):
                    X_batch = X[i:i + batch_size]
                    y_batch = y[i:i + batch_size]
                    output = self.forward(X_batch)
                    self.backward(X_batch, y_batch, output)

        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(thread_id, X, y))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # Return the final weights after training
        return self.W1, self.W2

# Sample usage
input_size = 10
hidden_size = 5
output_size = 2
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Generate dummy data
X = np.random.randn(100, input_size)
y = np.random.randn(100, output_size)

# Train the network
epochs = 10
batch_size = 10
num_threads = 4
W1, W2 = nn.train_multithreaded(X, y, epochs, batch_size, num_threads)

print("W1:", W1)
print("W2:", W2)
