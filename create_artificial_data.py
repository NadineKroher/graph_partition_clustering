import random
import numpy as np
import matplotlib.pyplot as plt
import math

N = 200
def point(h, k, r, rr):
    theta = np.random.random() * 2 * np.pi
    r += rr * random.random()
    return h + np.cos(theta) * r, k + np.sin(theta) * r


data_a = np.asarray([point(0, 0, 0.1, 0.1) for _ in range(N)])
data_b = np.asarray([point(0, 0, 0.5, 0.2) for _ in range(N)])
labels = np.zeros((2 * N, 1))
labels[N:] = 1

# plots
plt.figure(figsize=(7,6))
plt.scatter(data_a[:,0], data_a[:, 1])
plt.scatter(data_b[:,0], data_b[:, 1])
plt.show()

data = np.vstack((data_a, data_b))
print(data.shape, labels.shape)
data = np.hstack((data, labels))

np.savetxt('./data/artificial.txt', data)
