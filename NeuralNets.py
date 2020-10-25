import numpy as np


class Perceptron:
    def __init__(self):
        np.random.seed(10)
        self.weights_h = np.random.uniform(low=-1, high=1, size=(784, 2))
        self.weights_o = np.random.uniform(low=-1, high=1, size=(2, 1))
        self.bias_h = np.random.uniform(low=-1, high=1, size=(2,))
        self.bias_o = np.random.uniform(low=-1, high=1, size=(1,))
        self.alpha = 0.5
        self.epochs = 1

    def train(self, data):
        for i in range(self.epochs):
            for j in range(len(data)):
                # perform backprop for output
                x = data[j]
                output = self.forward_prop(x)
                raw_error = data[j, 0] - output
                delta_output = raw_error * output * (1 - output)
                # backprop error to hidden layer
                input_layer = x[1:]
                hidden_layer = self.activate_layer(np.dot(self.weights_h.transpose(), input_layer) + self.bias_h)  # generate hidden layer
                delta_hidden = np.multiply(np.multiply(hidden_layer, np.subtract(1, hidden_layer)), np.dot(self.weights_o, delta_output))
                # update output and internal weights
                for i in range(len(self.weights_o)):
                        self.weights_o[i] = self.weights_o[i] + self.alpha * hidden_layer[i] * delta_output
                for row in range(len(self.weights_h)):
                    for col in range(len(self.weights_h[row])):
                        self.weights_h[row, col] = self.weights_h[row, col] + self.alpha * input_layer[row] * delta_hidden[col]
                # update bias weights
                self.bias_o[0] = self.bias_o[0] + self.alpha * delta_output
                for row in range(len(self.bias_h)):
                    self.bias_h[row] = self.bias_h[row] + self.alpha * delta_hidden[row]

    def forward_prop(self, x):
        # x is 1x785 flat image with class label at index 0 (original 28x28 image)
        input_layer = x[1:]
        hidden_layer = self.activate_layer(np.dot(self.weights_h.transpose(), input_layer) + self.bias_h)
        output = self.activate_layer(np.dot(self.weights_o.transpose(), hidden_layer) + self.bias_o)
        return output

    def activate_layer(self, layer):
        # activation uses sigmoid
        activation = np.zeros(np.shape(layer))
        for i in range(len(layer)):
            activation[i] = 1/(1 + np.exp(-layer[i]))
        return activation

    def accuracy(self, data):
        correct = 0
        for row in range(len(data)):
            prediction = self.forward_prop(data[row])
            prediction = round(prediction[0])
            if prediction == data[row, 0]:
                correct += 1
        return correct/len(data)


def test_perceptron():
    train_data = np.genfromtxt('data/mnist_train_0_1.csv', delimiter=',')
    y = train_data[:, 0:1]
    train_data = np.divide(train_data[:, 1:], 255)
    train_data = np.c_[y, train_data]
    nn = Perceptron()
    nn.train(train_data)

    test_data = np.genfromtxt('data/mnist_test_0_1.csv', delimiter=',')
    y = test_data[:, 0:1]
    test_data = np.divide(test_data[:, 1:], 255)
    test_data = np.c_[y, test_data]
    print(nn.accuracy(test_data))


test_perceptron()
