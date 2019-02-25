import numpy as np

from compute_cost import compute_cost
from gradient_descent import gradient_descent
from load_data import load_data
from plot_data import plot_data, plot_cost
from predict import predict


# Linear Regression: Company Profit per City Population


# Load File and Set Initial Parameters

filename = 'city_profit.txt'
x, y = load_data(filename)
size = y.size
theta = np.array([[0.0], [0.0]])
alpha = 0.01
iterations = 1500
population = 175000


# Cost Function

cost = compute_cost(x, y, size, theta=theta)
print(f"With given theta: \n\tCost computed = {cost}\n")


# Gradient Descent

new_theta, cost_history = gradient_descent(x, y, size, theta=theta, alpha=alpha,
                                           iterations=iterations)


# Plot Data and Regression Line

plot_data(x, y, new_theta)


# Plot Cost History

plot_cost(iterations, cost_history)


# Make a Prediction

predict(population, new_theta)
