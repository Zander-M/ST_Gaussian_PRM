"""
    code tests
"""
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np

def test_gaussian():
    """
        Test Gaussian KDE
    """
    # Generate some random data
    data = np.random.normal(0, 1, size=1000)

    # Fit a Gaussian KDE to the data
    kde = gaussian_kde(data)

    # Create a grid of points where we want to evaluate the KDE
    x_grid = np.linspace(-5, 5, 1000)

    # Evaluate the KDE on the grid
    kde_values = kde(x_grid)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, kde_values, label='Gaussian KDE')
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram of data')
    plt.title('Gaussian Kernel Density Estimate')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("test.png")

if __name__ == "__main__":
    test_gaussian()