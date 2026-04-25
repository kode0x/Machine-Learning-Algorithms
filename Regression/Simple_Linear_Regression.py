import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None      # Slope
        self.intercept_ = None # Intercept

    def fit(self, X, y):
        """
        Fit Linear Model

        Parameters
        ----------
        X : Array-Like, Shape (n_samples,)
        y : Array-Like, Shape (n_samples,)
        """
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("Cannot compute slope because variance of X is zero")

        self.coef_ = numerator / denominator
        self.intercept_ = y_mean - self.coef_ * x_mean

        return self

    def predict(self, X):
        """
        Predict Using The Linear Model

        Parameters
        ----------
        X : Array-Like, Shape (n_samples,)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        X = np.asarray(X).reshape(-1)
        return self.intercept_ + self.coef_ * X
