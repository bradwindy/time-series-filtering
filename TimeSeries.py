import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian


if __name__ == '__main__':
    # Initialise some variables related to the time series
    # The number of values in the time series
    time_series_value_amount = 1000

    # Number of oscillations of the sin wave
    length = 10

    # Generating X values for the time series
    x = np.linspace(0, length, time_series_value_amount)

    # Generating noise for the data
    noise = np.random.normal(scale=0.5, size=time_series_value_amount)

    # Generating a sine wave for the data
    sin_wave = np.sin(2 * np.pi * x)

    # Combining the sine wave with the noise to form the Y data, which can then be filtered.
    y = sin_wave + noise

    # Plotting the line of the sine wave
    plt.plot(x, sin_wave)

    # Plotting the noisy Y data
    plt.plot(x, y, ls='none', marker='.', color='k')

    # Gaussian filtering of time series data
    b = gaussian(40, 8)
    gauss = filters.convolve1d(y, b / b.sum())

    # Print the deviation of the filtered data from a true sine wave
    print("Root Mean Square deviation of Gaussian Filtering: ",
          np.sqrt(np.sum(np.power(sin_wave - gauss, 2))) / time_series_value_amount)

    # Plotting gaussian filtered data, adding a legend to the plot and saving the figure
    plt.plot(x, gauss)
    plt.legend(['True Sin Wave', 'Measured Values', 'Gaussian Filtering'], loc='upper center')
    plt.savefig("plot.png")
