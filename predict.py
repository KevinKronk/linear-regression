import numpy as np


def predict(population, theta):
    """ Predicts the company profit based on the given city population and parameters. """

    prediction = int((np.dot([1, population / 10000], theta)) * 10000)
    print(f"For the given city population: \n\tThe predicted profit will be ${prediction}\n")
