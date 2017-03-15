import tensorflow as tf
import numpy as np
from scipy.special import expit  # expit = logistic function

#  tuneable values
GAPS_SEGMENT_LENGTH = 4000
BASES_SEGMENT_LENGTH = 1000


class AutoEncoder(object):
    def __init__(self, inputSize, hiddenSize, randomSeed=28):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        np.random.seed(randomSeed)
        self.encoderWeights = np.random.randn(self.inputSize, self.hiddenSize).astype(np.float32)
        np.random.seed(randomSeed)
        self.decoderWeights = np.random.randn(self.hiddenSize, self.inputSize).astype(np.float32)

    def encode(self, data):
        """
        Encodes data using trained autoencoder weights using numpy (not TensorFlow)
        :param data: Data to be encoded
        :return: Encoded data
        """
        return expit(np.dot(data, self.encoderWeights))

    def decode(self, data):
        """
        Encodes data using trained autoencoder weights using numpy (not TensorFlow).
        AutoEncoder.train() MUST be run before running this function.
        :param data:
        :return:
        """
        return expit(np.dot(data, self.decoderWeights))

    def train(self, data, epochs):
        """
        Trains autoencoder to fit the data using TensorFlow
        loosely based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
        :param data: array of batches of data to be fit
        :param epochs: number of epochs to run training
        :return: nothing
        """
        # Convert weights to TensorFlow variable
        encoderWeightsTF = tf.Variable(self.encoderWeights)
        decoderWeightsTF = tf.Variable(self.decoderWeights)

        x = tf.placeholder(np.float32, [None, self.inputSize])  # placeholder for training data

        encoded = tf.nn.sigmoid(tf.matmul(x, encoderWeightsTF))
        decoded = tf.nn.sigmoid(tf.matmul(encoded, decoderWeightsTF))

        costFunction = tf.reduce_mean(tf.pow(x - decoded, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(costFunction)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in xrange(epochs):
                for batch in data:
                    _, cost = sess.run([optimizer, costFunction], feed_dict={x: batch})
                if (epoch + 1) % 500 == 0:
                    print "Epoch: {0}, Cost: {1}".format(epoch + 1, cost)
            self.encoderWeights = encoderWeightsTF.eval()
            self.decoderWeights = decoderWeightsTF.eval()


class EncodedData(object):
    __slots__ = ('encoded', 'minValue', 'maxValue')

    def __init__(self, encoded, minValue, maxValue):
        self.encoded = encoded
        self.minValue = minValue
        self.maxValue = maxValue



def findErrorMatrix(input, reconstructedInput):
    """
    Creates the error matrix used in decompression
    :param input: original matrix
    :param reconstructedInput: matrix, having  been reconstructed after compression
    :return: error matrix containing position and difference of each inconsistency between the two matrices
    """
    errorMatrix = [[]]
    errorMatrixOriginal = input - reconstructedInput
    for i in xrange(errorMatrixOriginal.shape[0]):
        for j in xrange(errorMatrixOriginal.shape[1]):
            if errorMatrixOriginal[i,j] != 0:
                errorMatrix.append([i, j, errorMatrixOriginal[i, j]])

    return errorMatrix


def simplify(OGComp, zeroThreshold=0.001, oneThreshold=0.99):
    """
    Rounds numbers that are very close to 0 or 1
    :param OGComp: Matrix being processed
    :param zeroThreshold: Threshold value before an element is rounded to 0
    :param oneThreshold: Threshold value before an element is rounded to 1
    :return:
    """
    processedComp = OGComp.copy()
    for i in xrange(OGComp.shape[0]):
        for j in xrange(OGComp.shape[1]):
            if OGComp[i, j] < zeroThreshold:
                processedComp[i, j] = 0
            if OGComp[i, j] > oneThreshold:
                processedComp[i, j] = 1

    return processedComp


def normalize(data):
    """
    Normalizes data to between 0 and 1
    :param data: data to be normalized
    :return: normalized data
    """
    return (data - data.min()) / (data.max() - data.min())

def denormalize(data, minValue, maxValue):
    """
    Inverse of normalize()
    :param data: data to be denormalized
    :return:
    """
    return (data * (maxValue - minValue)) + minValue