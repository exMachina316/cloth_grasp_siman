import pandas as pd

import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('optimization_results.csv')

# Plot the data
plt.figure(figsize=(10, 6))

# plt.plot(data['Iteration'], data['Temperature'], label='Temperature')
plt.plot(data['Iteration'], data['Best Fitness'], label='Best Fitness')
plt.plot(data['Iteration'], data['Curvature'], label='Curvature')
plt.plot(data['Iteration'], data['Density'], label='Density')
plt.plot(data['Iteration'], data['Strain'], label='Strain')


plt.xlabel('Iteration')
plt.ylabel('Values')
plt.title('Optimization Results')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()