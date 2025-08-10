import unittest
import KNN
import numpy as np

class TestKNN(unittest.TestCase):
    def test_get_euclidean_distance(self):
        result1D = KNN.get_euclidean_distance(np.array([1]), np.array([4]))
        result2D = KNN.get_euclidean_distance(np.array([3,2]), np.array([4,1]))
        result4D = KNN.get_euclidean_distance(np.array([2,3,4,5]), np.array([5,4,3,2]))

        self.assertEqual(result1D, 3)
        self.assertEqual(result2D, np.sqrt(2))
        self.assertEqual(result4D, np.sqrt(20))

if __name__ == "__main__":
    unittest.main()