import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

w1 = np.random.rand(2, 2)
w2 = np.random.rand(1, 2)

b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

print(w1)
print(w2)
print(b1)
print(b2)

lr = 0.5

for epochs in range(10000):
    z1 = np.dot(X, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.dot(a1, w2.T) + b2
    a2 = 1 / (1 + np.exp(-z2))

    error = a2 - target
    dz2 = error * (a2 * (1 - a2))
    dw2 = np.dot(a1.T, dz2)

    dz1 = np.dot(dz2, w2) * (a1 * (1 - a1))
    dw1 = np.dot(X.T, dz1)

    w2 -= lr * dw2.T
    w1 -= lr * dw1

    b2 -= lr * np.sum(dz2, axis=0, keepdims=True)
    b1 -= lr * np.sum(dz1, axis=0, keepdims=True)

print("Predictions after training:")
print(a2)