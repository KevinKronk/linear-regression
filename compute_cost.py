import numpy as np


def compute_cost(x, y, size, theta):
    """
        Compute the cost function for linear regression.

        Parameters
        ----------
        x : array_like
            Shape (m, n+1), where m is the number of examples, and n is the number of features
            including the vector of ones for the zeroth parameter.

        y : array_like
            Shape (m,), where m is the value of the function at each point.

        size : int
            Number of total training points.

        theta : array_like
            Shape (n+1, 1). Starting parameters of the regression function.

        Returns
        -------
        cost : float
            The value of the regression cost function.
    """

    cost = np.sum((1 / (2 * size)) * (((np.dot(theta.T, x)) - y) ** 2))
    return cost
