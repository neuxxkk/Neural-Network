import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scp
import scipy
import subprocess
import time
import sys

class neuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learn_rate

        self.activation_function = lambda x: scp.expit(x)

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
    

#INICIATING

print(6 * '=-', "AIemao 10 epoch clock", 6 * '=-')

print("\nTreinando rede neural", end='', flush=True)

#DEFINE

#nodes
INPUT_NODES = 784
OUTPUT_NODES = 10
HIDDEN_NODES = 200

#learn factor
LEARNING_FACTOR = 0.01

#Discount Factor
DISCOUNT_FACTOR = .0

#Instance
n = neuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_FACTOR)

train_file = open("train-1000.csv")

plot_numb = []
tests = []

epoch = 5

for __ in range(epoch):
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
        #normal
        n.train(inputs, targets)
        
        #clockwise 10
        inputs_plus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), 10,cval=0.01, reshape=False)
        n.train(inputs_plus10_img.reshape(784), targets)

        #-10
        inputs_minus10_img = scipy.ndimage.rotate(inputs.reshape(28,28), -10, cval=0.01, reshape=False)
        n.train(inputs_minus10_img.reshape(784), targets)

    LEARNING_FACTOR -= DISCOUNT_FACTOR


# Testing

opt = 1

while opt:

    opt = int(input('\n\nEcolha uma opção: \n1 - Um dígito da folha de testes \n2 - Teste de desempenho \n3 - Desenhe você mesmo \n0 - Quit \n: '))

    match opt:

        case 1:
            test_file = open("test-10.csv")

            idx = int(input("\nIndex: "))

            # reading testes sample 4
            one_digit = np.asfarray(test_file.readlines()[idx].split(','))[1:]

            #plotting chosen number
            plt.imshow(one_digit.reshape(28,28), cmap='Greys')
            plt.show()

            #showing Network
            answer = n.query((np.asfarray(one_digit) / 1 * 0.99) + 0.01, True)

            # SHOW ANSWER
            #print(answer)
            print("The number is: ", np.argmax(answer))
            time.sleep(1.5)


        case 2:
            # test the neural network
            test_file = open("train-1000.csv")

            # scorecard for how well the network performs, initially empty
            scorecard = []

            # go through all the records in the test data set
            for record in test_file:
                # split the record by the ',' commas
                all_values = record.split(',')
                # correct answer is first value
                correct_label = int(all_values[0])
                # scale and shift the inputs
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # query the network
                outputs = n.query(inputs)[1]
                # the index of the highest value corresponds to the label
                label = np.argmax(outputs)

                """
                plt.imshow(np.asfarray(all_values[1:]).reshape(28,28), cmap='Greys')
                plt.show()
                print(label) 
                """

                # append correct or incorrect to list
                if (label == correct_label):
                    # network's answer matches correct answer, add 1 to scorecard
                    scorecard.append(1)
                else:
                    # network's answer doesn't match correct answer, add 0 to scorecard
                    scorecard.append(0)
                    pass
                
                pass
            # calculate the performance score, the fraction of correct answers
            scorecard_array = np.asarray(scorecard)
            print ("Performance = ", scorecard_array.sum() / scorecard_array.size)
            time.sleep(1.5)
            

        case 3:
            file_test = open('number.csv', 'r')

            subprocess.run(['./draw_table/linux-amd64/draw_table'])


            inputs = (np.asfarray(file_test.readline().split(',')) / 255 * 0.99) + 0.01

                        
            """             
            plt.imshow(inputs.reshape(28,28), cmap='Greys')
            plt.show() 
            """
           

            print("The Number is: ", np.argmax(n.query(inputs)[1]))
            time.sleep(1.5)

        case _: quit()

