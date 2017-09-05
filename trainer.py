import numpy as np
import time


class Trainer:
    no_of_layers = 0;
    shape = None;
    weights = [];

    def __init__(self, length):
        layer = (260, 25, 25, length)
        self.no_of_layers = len(layer) - 1;
        self.shape = layer

        self.input_layer = []
        self.output_layer = []
        self.previous_delta_weights = []

        for (l1, l2) in zip(layer[:-1], layer[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(l2, l1 + 1)))
            self.previous_delta_weights.append(np.zeros((l2, l1 + 1)))

        print len(self.weights[2])

    def forwardProc(self, input):
        input_cases = input.shape[0]
        print self.weights[1].shape
        self.input_layer = []
        self.output_layer = []
        for index in range(self.no_of_layers):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, input_cases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self.output_layer[-1], np.ones([1, input_cases])]))
            self.input_layer.append(layerInput)
            self.output_layer.append(self.sigmoid(layerInput))
        return self.output_layer[-1].T

    def train(self, input, target, trainingRate=0.2, momentum=0.5):
        delta = []
        input_cases = input.shape[0]
        self.forwardProc(input)
        for index in reversed(range(self.no_of_layers)):
            if index == self.no_of_layers - 1:
                output_delta = self.output_layer[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.sigmoid(self.input_layer[index], True))
            else:
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.sigmoid(self.input_layer[index], True))

        for index in range(self.no_of_layers):
            delta_index = self.no_of_layers - 1 - index
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, input_cases])]) #260 X 70 . 1 X 70
            else:
                layerOutput = np.vstack([self.output_layer[index - 1], np.ones([1, self.output_layer[index - 1].shape[1]])])

            current_delta_weight = np.sum(layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0), axis=0)
            delta_weight = trainingRate * current_delta_weight + momentum * self.previous_delta_weights[index]
            self.weights[index] -= delta_weight
            self.previous_delta_weights[index] = delta_weight

        return error

    def sigmoid(self, x, Derivative=False):
        if not Derivative:
            return 1 / (1 + np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out * (1 - out)


if __name__ == "__main__":

    with open('words.txt') as f:
        words = f.read().splitlines()

    trn = Trainer(len(words))
    f = []
    inputarray = []
    input = []
    for i in range(0, len(words)):
        f = open("mfcc/" + words[i]+ "_mfcc.npy")
        input.append(np.load(f))
        inputarray=np.concatenate(input)

    print inputarray.shape

    a = np.zeros((len(words), len(words)), int)
    np.fill_diagonal(a, 1)
    targetarray= []
    for i in range(0,len(words)):
        t=np.array([a[i] for _ in range(len(input[i]))])
        targetarray.append(t)

    target2=np.concatenate(targetarray)
    print (target2.shape)

    lnMax = 1000000
    lnErr = 1e-5


    for i in range(lnMax - 1):
        err = trn.train(inputarray, target2, momentum=0.3)
        if i % 1500 == 0:
            print("Iteration {0} \tError: {1:0.6f}".format(i, err))
        if err <= lnErr:
            print("Minimum error reached at iteration {0}".format(i))
            break


    with open("weights/" + "words" + ".npy", 'w') as outfile:
        np.save(outfile, trn.weights)


