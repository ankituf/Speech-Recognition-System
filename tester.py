from __future__ import division
import numpy as np
import scipy.io.wavfile as wav
from mfcc_features import mfcc


class Tester:
    no_of_layers = 0;
    shape = None;
    weights = [];

    def __init__(self, length, weights):

        layer = (260, 25, 25, length)
        self.no_of_layers = len(layer) - 1;
        self.shape = layer

        self.input_layer = []
        self.output_layer = []
        self.weights = weights

    def forwardProc(self, input):
        InCases = input.shape[0]
        print input.shape[1]
        print self.weights[0].shape
        self.input_layer = []
        self.output_layer = []

        for index in range(self.no_of_layers):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, InCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self.output_layer[-1], np.ones([1, InCases])]))
            self.input_layer.append(layerInput)
            self.output_layer.append(self.sigmoid(layerInput))
        print self.output_layer[-1].T
        return self.output_layer[-1].T

    def sigmoid(self, x, Derivative=False):
        if not Derivative:
            return 1 / (1 + np.exp(-x))
        else:
            out = self.sigmoid(x)
            return out * (1 - out)


def testInit():
    f1 = open("weights/words.npy", "rb")
    weights = np.load(f1)
    with open('words.txt') as f:
        words = f.read().splitlines()
    net = Tester(len(words), weights)
    return net


def extractFeature(soundfile):
    (rate, sig) = wav.read(soundfile)
    duration = len(sig) / rate;
    mfcc_features = mfcc(sig, rate, winlen=duration / 20, winstep=duration / 20)
    m = mfcc_features[:20]
    mfcc_arr = []
    for i in m:
        mfcc_arr.extend(i)
    mfcc_arr /= np.max(np.abs(mfcc_arr), axis=0)
    inputArray = np.array([mfcc_arr])
    return inputArray


def feedToNetwork(inputArray, testNet):
    outputArray = testNet.forwardProc(inputArray)
    indexMax = outputArray.argmax(axis=1)[0]

    with open('words.txt') as f:
        words = f.read().splitlines()
    outStr = "Detected: ",words[indexMax]
    print "Detected: ",words[indexMax]
    return outStr


if __name__ == "__main__":
    print ''
    # testNet = testInit()
    # inputArray = extractFeature("test_files/test.wav")
    # feedToNetwork(inputArray, testNet)
