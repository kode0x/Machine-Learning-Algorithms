import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None # Slope
        self.intercept_ = None # Intercept

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        # Add Bias Column (Intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal Equation
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

        return self

    def predict(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_