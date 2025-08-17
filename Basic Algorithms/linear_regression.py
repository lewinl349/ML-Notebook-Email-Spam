import numpy as np

class LR:
    def __init__(self, learn_rate, epochs):
        # Support multiple weights/variables
        self.weights = None
        self.bias = 0.5
        self.learn_rate = learn_rate
        self.epochs = epochs

        # Keep track of the losses as extra data
        self.losses = []

        return

    def fit(self, X, y):
        """
        Find the line with the training data X (features), y (label)

        Parameters
        -------------
        X: np.ndarray
        y: np.float64
        """

        samples, num_of_weights = len(X), len(X[0])
        self.weights = np.array([0.5 for _ in range(num_of_weights)])

        for _ in range(self.epochs):
            # Find MSE
            y_preds = np.dot(X, self.weights) + self.bias
            errors = (y - y_preds)
            mse = np.mean(errors ** 2)
            self.losses.append(mse)

            # Gradient descent
            w_slope = -2 * (np.dot(X.T, errors) / samples)
            b_slope = -2 * np.mean(errors)

            self.weights -= self.learn_rate * w_slope
            self.bias -= self.learn_rate * b_slope
        
        return
    
    def predict(self, X):
        """
        For the given features (x1, ..., xn), find the label (y)
        y = b + w1 * x1 + ... + wn * xn

        Parameters
        -------------
        X: np.ndarray

        returns np.float64
        """
        y_pred = self.bias + np.dot(X, self.weights) 

        return y_pred
    
    def get_losses(self):
        """
        Returns the array of losses of each epoch

        returns array(np.float64)
        """
        return self.losses