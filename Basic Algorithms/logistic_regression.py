import numpy as np

def sigmoid(z):
    """
    Return sigmoid function with a given z in (e^z)

    Parameters
    -------------
    z: np.float64

    returns np.float64
    """

    return (1 / (1 + np.exp(-z)))

class LogR:
    def __init__(self, learn_rate, epochs):
        self.weights = None
        self.bias = 0.1
        self.learn_rate = learn_rate
        self.epochs = epochs

        # Keep track of the losses as extra data
        self.losses = []

    def fit(self, X, y):
        """
        Find sigmoid function with the training data X (features), y (label)

        Parameters
        -------------
        X: np.ndarray - X_train
        y: np.float64 - y_train
        """
        samples, num_of_weights = X.shape
        self.weights = np.array([0.1 for _ in range(num_of_weights)])

        for _ in range(self.epochs):
            # Find MSE
            z = np.dot(X, self.weights) + self.bias
            y_preds = sigmoid(z)
            errors = (y_preds - y)

            # Prevent log (0) errors by adding 1e-12
            log_loss = -np.mean(y * np.log(y_preds + 1e-12) + (1 - y) * np.log(1 - y_preds + 1e-12))
            self.losses.append(log_loss)

            # Gradient descent
            w_slope = (np.dot(X.T, errors) / samples)
            b_slope = np.mean(errors)

            self.weights -= self.learn_rate * w_slope
            self.bias -= self.learn_rate * b_slope
        

        return
    
    def predict(self, X):
        """
        For the given features (x1, ..., xn), find the label (y)
        Using sigmoid function

        Parameters
        -------------
        X: np.ndarray - X_test

        returns np.float64
        """
        z = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(z)

        return y_pred > 0.5
    
    def get_losses(self):
        """
        Returns the array of losses of each epoch

        returns array(np.float64)
        """
        return self.losses