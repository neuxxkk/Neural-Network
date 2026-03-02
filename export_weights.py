import json
import numpy as np
import scipy.special as scp
import scipy.ndimage

class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learn_rate
        self.activation_function = lambda x: scp.expit(x)
        self.Wih = np.random.normal(0, pow(self.inodes, -.5), (self.inodes, self.hnodes))
        self.Who = np.random.normal(0, pow(self.hnodes, -.5), (self.hnodes, self.onodes))

    def query(self, inputs):
        hidden_inputs = np.dot(inputs, self.Wih)
        hidden_outputs = np.array(self.activation_function(hidden_inputs), ndmin=2)
        final_inputs = np.dot(hidden_outputs, self.Who)
        final_outputs = np.array(self.activation_function(final_inputs), ndmin=2)
        return hidden_outputs, final_outputs

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2)
        targets = np.array(targets, ndmin=2)
        hidden_outputs, final_outputs = self.query(inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(output_errors, self.Who.T)
        self.Wih += self.lr * np.dot(np.transpose(inputs), hidden_errors * hidden_outputs * (1 - hidden_outputs))
        self.Who += self.lr * np.dot(np.transpose(hidden_outputs), output_errors * final_outputs * (1 - final_outputs))

print("Training neural network with train-1000.csv...")

INPUT_NODES = 784
OUTPUT_NODES = 10
HIDDEN_NODES = 200
LEARNING_FACTOR = 0.1
epoch = 7

n = neuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_FACTOR)

train_file = open("train-1000.csv")
lines = train_file.readlines()
train_file.close()

for e in range(epoch):
    print(f"Epoch {e+1}/{epoch}...")
    for idx, case in enumerate(lines):
        if idx % 200 == 0:
            print(f"  Sample {idx}/{len(lines)}")
        
        matrix_n = np.asarray(case.split(',')[1:], dtype=float)
        inputs = (matrix_n / 255 * 0.99) + 0.01
        targets = np.zeros(OUTPUT_NODES) + 0.01
        targets[int(case.split(',')[0])] = 0.99
        
        # Normal training
        n.train(inputs, targets)
        
        # +10 degree rotation
        inputs_plus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), 10, cval=0.01, reshape=False)
        n.train(inputs_plus10_img.reshape(784), targets)
        
        # -10 degree rotation
        inputs_minus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), -10, cval=0.01, reshape=False)
        n.train(inputs_minus10_img.reshape(784), targets)

print("Training complete! Exporting weights...")

# Export weights to JSON
weights = {
    "Wih": n.Wih.tolist(),
    "Who": n.Who.tolist()
}

with open("weights.json", "w") as f:
    json.dump(weights, f)

with open("weights.js", "w") as f:
    f.write(f"const pretrainedWeights = {json.dumps(weights)};")

print(f"Weights exported to weights.json and weights.js (Wih: {n.Wih.shape}, Who: {n.Who.shape})")
