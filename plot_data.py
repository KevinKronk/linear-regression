from matplotlib import pyplot as plt
import numpy as np


def plot_data(x, y, theta):
    """
        Plots the data and the linear regression line.

        Parameters
        ----------
        x : array_like
            Shape (m, n+1), where m is the number of examples, and n is the number of features
            including the vector of ones for the zeroth parameter.

        y : array_like
            Shape (m,), where m is the value of the function at each point.

        theta : array_like
            Shape (n+1, 1). Optimized linear regression parameters.
    """

    # Plots the data without the vector of ones for the zeroth parameter
    plt.plot(x[1, :], y, 'bo', ms=10, mec='k')

    # Creates y axis for the linear regression line
    new_y = np.dot(theta.T, x)
    plt.plot(x[1, :], new_y.T, '-', color='orange', linewidth=3)

    plt.title("Corporate Profit per City Population")
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    plt.legend(['Training Data', 'Linear Regression'])
    plt.show()


def plot_cost(iterations, cost_history):
    """
        Plots the cost over each iteration of gradient descent.

        Parameters
        ----------
        iterations : int
            The number of iterations for gradient descent.

        cost_history : list
            A list of the values of the cost function after each iteration.
    """

    plt.plot(range(iterations), cost_history, linewidth=3)
    plt.title("Cost History")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
