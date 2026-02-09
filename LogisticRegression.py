
import numpy as np

class LogisticRegression:
    def __init__(self, x_data, y_data):
        self.X = np.array(x_data, dtype=float)
        self.y = np.array(y_data, dtype=float)
        self.beta = None        # β₁ … βₙ
        self.beta_0 = 0.0       # β₀

    def sigmoid(self, X, beta, beta_0):
        z = np.dot(X, beta) + beta_0
        return 1 / (1 + np.exp(-z))
    
    def fit(self, lr=0.1, epochs=2000):
        X = self.X
        y = self.y

        number_of_samples, number_of_features = X.shape

        self.beta = np.zeros(number_of_features)
        self.beta_0 = 0.0

        for _ in range(epochs):

            p = self.sigmoid(X, self.beta, self.beta_0)

            error = p - y

            # ∂L/∂β = (1/m) Xᵀ(p − y)
            beta_gradients = (1 / number_of_samples) * np.dot(X.T, error)

            # ∂L/∂β₀ = (1/m) Σ(p − y)
            beta_0_gradient = (1 / number_of_samples) * np.sum(error)

            self.beta = self.beta - lr * beta_gradients
            self.beta_0 = self.beta_0 - lr * beta_0_gradient

    def predict(self, x_new):
        x_new = np.array(x_new, dtype=float)

        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)

        probabilities = self.sigmoid(x_new, self.beta, self.beta_0)
        return (probabilities >= 0.5).astype(int)
