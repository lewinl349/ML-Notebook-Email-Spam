import numpy as np
from collections import Counter

def get_euclidean_distance(x1, x2):
    """ Can be in any number of dimensions
    Ex:
    get_euclidean_distance(numpy.array([3, 2]), numpy.array([1, 2]))
    get_euclidean_distance(numpy.array([1, 5, 4]), numpy.array([1, 3, 2]))

    Parameters
    -------------
    x1: numpy.ndarray
    x2: numpy.ndarray

    returns numpy.float64
    """

    dist = np.sqrt(np.sum((x2 - x1) ** 2)) 
    return dist

class KNN:
    def __init__(self, k=3):
        """
        Parameters
        -------------
        k : int
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Store the training data

        Parameters
        -------------
        X_train : numpy.ndarray
        y_train : numpy.ndarray
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Go through every point in the testing sample

        Parameters
        -------------     
        X_test: np.ndarray
        """
        preds = [self._predict(x) for x in X_test]
        return preds

    def _predict(self, x):
        """
        Helper method to predict an individual point

        Parameters
        -------------
        x: np.ndarray
        """

        # 1. Find distance from every point in the training sample
        distances = [get_euclidean_distance(x, train) for train in self.X_train] 

        # 2. Find the closest k points and their y label
        # argsort() returns the indices of the sorted array
        # [20, 45, 32] -> [0, 2, 1]
        closest_k_indices = np.argsort(distances)[:self.k]
        y_outcomes = [self.y_train[i] for i in closest_k_indices]

        # 3. Choose the y label that occurs the most
        # most_common() returns [('y', 5),('b', 3),('c', 2)]
        outcome = Counter(y_outcomes).most_common()
        return outcome[0][0]