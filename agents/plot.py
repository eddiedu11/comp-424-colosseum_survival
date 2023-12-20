import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Sample data
x_data = np.array([4, 6, 8, 10, 12, 14, 16])
y_data = np.array([20, 45, 60, 174, 289, 460, 467])

# Define the model function (e.g., quadratic)
def model_func(x, a, b, c):
    return a * x**2 + b * x + c

# Perform curve fitting
params, covariance = curve_fit(model_func, x_data, y_data)

# Extract parameters
a, b, c = params

# Generate fitted curve
fitted_curve = model_func(x_data, a, b, c)

# Plot original data and fitted curve
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_data, fitted_curve, label='Fitted Curve', color='red')

# Add axis titles
plt.xlabel('Board_Size')
plt.ylabel('Max Breadth')

plt.legend()
plt.show()
