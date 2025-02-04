import matplotlib.pyplot  as plt
import numpy as np
import os

f = open("/home/vitornms/100-train.csv")

number_graphs = []
inputs = []

for n in f.readlines():
    # transform text line (CSV) into number in matrix 28x28 (784 pixels)
    matrix_n = np.asfarray(n.split(',')[1:]).reshape(28,28)
    number_graphs.append(matrix_n)

    # convert 0 - 255 into 0.01 -  1.00
    inputs.append((matrix_n / 255 * 0.99) + 0.01)

print(inputs[0])

""" plt.imshow(number_graphs[0], cmap='Greys')
plt.show() """