import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Set the random seed for reproducibility
np.random.seed(0)

# Define the number of observations
n = 1000

# Generate the time-varying parameters
alpha0 = np.linspace(0.2, 0.8, n)
alpha1 = np.sin(np.linspace(0, 4 * np.pi, n))

# Simulate the time series
eps = np.random.randn(n)
returns = np.zeros(n)
volatility = np.zeros(n)

for t in range(1, n):
    returns[t]= alpha0[t] + alpha1[t] * returns[t-1] + eps[t]
    volatility[t] = 0.01 + 0.1 * volatility[t-1] + 0.8 * np.abs(volatility[t-1]) * eps[t]

model = arch_model(returns)
model_fit = model.fit(disp='off')
print(model_fit.params)

plt.figure(figsize=(10, 6))
plt.plot(returns, label='Returns')
plt.xlabel('Time')
plt.show()
