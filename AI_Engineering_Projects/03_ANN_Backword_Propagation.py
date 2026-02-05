from os import error

import numpy as np

x = 2.0 #input
y_target = 10 #The answer we want
weight = 3.0
learning_rate = 0.1

print(f"--- STARTING ---")
print(f"Input: {x}, Target: {y_target}")
print(f"Initial Weight: {weight}\n")

for round in range(2):

    prediction = x * weight

    error = prediction - y_target

    gradient = 2 * error * x

    weight = weight - (learning_rate * gradient)

    print(f"Round {round + 1}:")
    print(f"   Prediction: {prediction:.2f}")
    print(f"   Error: {error:.2f}")
    print(f"   Gradient (The Nudge): {gradient:.2f}")
    print(f"   New Weight: {weight:.2f}\n")

print(f"--- FINAL RESULT ---")
print(f"Input {x} * Final Weight {weight:.2f} = {x * weight:.2f}")

X = np.array([[2.0,3.0]])

y_target = np.array([13.0])

weights = np.array([[1.0],[1.0]])

learning_rate = 0.01

print(f"Inputs: {X} (Shape: {X.shape})")
print(f"Initial Weights:\n{weights} (Shape: {weights.shape})\n")

for i in range(3):
    print(f"--- ROUND {i+1} ---")

    prediction = np.dot(X,weights)

    error = prediction - y_target

    gradient = 2 * np.dot(X.T,error)

    weights = weights - (learning_rate * gradient)

    print(f"Prediction: {prediction[0][0]:.2f}")
    print(f"Error: {error[0][0]:.2f}")
    print(f"Gradients (The Nudges):\n{gradient.T}")  # Printed sideways to save space
    print(f"New Weights:\n{weights}\n")