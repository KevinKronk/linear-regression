import numpy as np

from compute_cost import compute_cost


def gradient_descent(x, y, size, theta, alpha, iterations):
    """
        Performs gradient descent to optimize the 'theta' parameters. Updates theta for a total of
        inputted 'iterations', with a learning rate 'alpha'.

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

        alpha : float
            The learning rate.

        iterations : int
            The number of iterations for gradient descent.

        Returns
        -------
        theta : array_like
            Shape (n+1, 1). The optimized linear regression parameters.

        cost_history : list
            A list of the values of the cost function after each iteration.
    """

    cost_history = []

    for i in range(iterations):
        temp_cost = compute_cost(x, y, size, theta)
        cost_history.append(temp_cost)

        delta = (1 / size) * ((np.dot(theta.T, x)) - y) * x
        delta2 = delta.sum(axis=1, keepdims=True)
        theta = (theta - (alpha * delta2))
    print(f"The new optimized parameters are: \n{theta}\n")
    return theta, cost_history
