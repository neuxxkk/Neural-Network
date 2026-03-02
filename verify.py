import json
import numpy as np
import scipy.special as scp

class NeuralNetwork:
    def __init__(self, wih, who):
        self.Wih = np.array(wih)
        self.Who = np.array(who)
        self.activation_function = lambda x: scp.expit(x)

    def query(self, inputs):
        hidden_inputs = np.dot(inputs, self.Wih)
        hidden_outputs = np.array(self.activation_function(hidden_inputs), ndmin=2)
        final_inputs = np.dot(hidden_outputs, self.Who)
        final_outputs = np.array(self.activation_function(final_inputs), ndmin=2)
        return final_outputs

with open('weights.json', 'r') as f:
    weights = json.load(f)

nn = NeuralNetwork(weights['Wih'], weights['Who'])

with open('test-10.csv', 'r') as f:
    lines = f.readlines()

for line in lines:
    values = line.split(',')
    label = int(values[0])
    inputs = (np.asarray(values[1:], dtype=float) / 255.0 * 0.99) + 0.01
    outputs = nn.query(inputs)
    pred = np.argmax(outputs)
    print(f"True: {label}, Pred: {pred}")
