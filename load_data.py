import numpy as np


def load_data(filename):
    """ Loads a csv file with only two values per line. """

    with open(filename) as data:
        x, y = [], []
        for line in data:
            line_x, line_y = line.split(',')
            x.append(float(line_x))
            y.append(float(line_y))

        x = np.array(x)
        y = np.array(y)
        size = y.size

        # Add vector of ones for the zeroth parameter
        x = np.stack([np.ones(size), x], axis=0)

        return x, y
