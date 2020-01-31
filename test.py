from sklearn.datasets import make_moons, load_iris # import function from the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class KNNClassifier(object):

    def __init__(self, max_dist=1.):
        """
        This is a constructor of the class. 
        Here you can define parameters (max_dist) of the class and 
        attributes, that are visible within all methods of the class.
        Parameters
        ----------
        max_dist : float
            Maximum distance between an object and its neighbors.
        """
        # Make this parameter visible in all methods of the class
        self.max_dist = max_dist


    def calculate_distances(self, X, one_x):
        dists = np.sqrt( np.sum( (X - one_x)**2, axis=1 ) )
        return dists


    def fit(self, X, y):
        """
        This method trains the KNN classifier
        Actualy, the KNN classifier has no training procedure.
        It just remembers data (X, y) that will be used for predictions.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        y : numpy.array, shape = (n_objects)
            1D array with the object labels. 
            For the classification labels are integers in {0, 1, 2, ...}.
        """
        ### Your code here
        pass
    def predict(self, X):
        """
        This methods performs labels prediction for new objects.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        Returns
        -------
        y_predicted : numpy.array, shape = (n_objects)
            1D array with predicted labels.
            For the classification labels are integers in {0, 1, 2, ...}.
        """
        # Create an empty list for predicted labels
        y_predicted = []

        for x_one in X:
            distance = self.calculate_distance(self.X_train, x_one)

            k_neighbors_indeces = distance[distance <= self.max_dist]
        ### Replace this line with your code:
        y_predicted = np.random.randint(0, 2, len(X))
        ### The end of your code

        return np.array(y_predicted) # return numpy.array 
    def predict_proba(self, X):
        """
        This methods performs prediction of probabilities of each class for new objects.
        Parameters
        ----------
        X : numpy.array, shape = (n_objects, n_features)
            Matrix of objects that are described by their input features.
        Returns
        -------
        y_predicted_proba : numpy.array, shape = (n_objects, n_classes)
            2D array with predicted probabilities of each class. 
            Example:
                y_predicted_proba = [[0.1, 0.9],
                                     [0.8, 0.2], 
                                     [0.0, 1.0], 
                                     ...]
        """
        # Create an empty list for predictions
        y_predicted_proba = []

        ### Replace these lines with your code:
        y_predicted_proba = np.random.rand(len(X), 2) # for 2 classes
        y_predicted_proba /= y_predicted_proba.sum(axis=1).reshape(-1, 1)
        ### The end of your code 
        return np.array(y_predicted_proba) # return numpy.array


X , y = make_moons(n_samples = 1000, noise = 0.2, random_state = 42)
print(X)

ans = np.array([[-0.112,  0.52 ],
                [ 1.143, -0.343]])
assert np.array_equal(np.round(X[:2], 3), ans), ('Check your solution.')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.5)

ans = np.array([[ 0.77 , -0.289],
                [ 0.239,  1.041]])
assert np.array_equal(np.round(X_train[:2], 3), ans), ('Check your solution.')



