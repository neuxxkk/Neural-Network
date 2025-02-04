import numpy as np
import math
import matplotlib.pyplot as plt

class neuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learn_rate

        # Mine Ws are not transposed
        self.Wih = np.random.normal(0, pow(self.inodes, -.5), (self.inodes, self.hnodes)) # range = 0 to 1 / sqrt(incomming links)
        self.Who = np.random.normal(0, pow(self.hnodes, -.5), (self.hnodes, self.onodes)) # 1/sqrt(x) = x⁻⁽¹/²)

    def query(self, inputs, show=False):
        # input layer -> hidden layer
        hidden_inputs = np.dot(inputs, self.Wih)
        hidden_outputs = np.array(self.activation_function(hidden_inputs), ndmin=2)

        # hidden layer -> final layer
        final_inputs = np.dot(hidden_outputs, self.Who)
        final_outputs = np.array(self.activation_function(final_inputs), ndmin=2)

        if show: return final_outputs

        return hidden_outputs, final_outputs

    def train(self, inputs, targets):

        inputs = np.array(inputs, ndmin=2)
        targets = np.array(targets, ndmin=2)

        # oututs from query
        hidden_outputs, final_outputs = self.query(inputs)

        # errors layer 3 and 2
        output_errors = targets - final_outputs
        hidden_errors = np.dot(output_errors, self.Who.T)

        #DW = lr × (Eₖ × sigmoid(Oₖ) × (1 - sigmoid(Oₖ)) · Oⱼᵀ) = lr × (Oⱼ · (....)ᵀ)ᵀ, weights are not transposed like tariq did

        # Adjusting Weights between layers input -> hidden
        self.Wih += self.lr * np.dot(np.transpose(inputs), hidden_errors * hidden_outputs * (1 - hidden_outputs))

        # Adjusting Weights between layers hidden -> output, opposite from gradient (derivada, inclinação)
        self.Who += self.lr * np.dot(np.transpose(hidden_outputs), output_errors * final_outputs * (1 - final_outputs))

        pass

    def activation_function(self, x):
        return 1 / (1 + math.e ** -x)

#DEFINE

#nodes
INPUT_NODES = 784
OUTPUT_NODES = 10
HIDDEN_NODES = 100

#learn factor
LEARNING_FACTOR = 0.3

#Instance
n = neuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_FACTOR)

train_file = open("train-100.csv")

plot_numb = []
tests = []

for case in train_file.readlines():
    # transform text line (CSV) into number in matrix 28x28 (784 pixels)
    matrix_n = np.asfarray(case.split(',')[1:])
    plot_numb.append(matrix_n.reshape(28,28))

    # convert 0 - 255 into 0.01 -  1.00
    inputs = (matrix_n / 255 * 0.99) + 0.01

    tests.append(inputs)

    # targets set
    targets = np.zeros(OUTPUT_NODES) + 0.01
    targets[int(case[0])] = 0.99

    #training
    n.train(inputs, targets)


# Testing
test_file = open("test-10.csv")

# reading testes
l1 = np.asfarray(test_file.readlines()[5].split(',')[1:])

#plotting chosen number
plt.imshow(l1.reshape(28,28), cmap='Greys')
plt.show()

#showing Network
answer = n.query((np.asfarray(l1) / 255 * 0.99) + 0.01, True)

# SHOW ANSWER
print(answer)
print("The number is: ", np.argmax(answer))

