import numpy as np
import matplotlib.pyplot as plt

n = 65
r = 5
R = 10

theta = np.linspace(0, 2*np.pi, 400)

plt.figure(figsize=(6,6))

for i in range(n):
    angle = 2*np.pi * i / n
    cx = R * np.cos(angle)
    cy = R * np.sin(angle)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    
    plt.plot(x, y, color='black', linewidth=2)

plt.gca().set_aspect('equal')
plt.axis('off')

plt.show()