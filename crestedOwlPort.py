import tensorflow as tf
import numpy as np
from scipy.special import expit  # expit = logistic function

def simplify(OGComp, zeroThreshold=0.001, oneThreshold=0.99):
    """
    equivalent to processedComp [NOT 100% SURE WHAT THIS DOES]
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

def readAndProcess(filePath):
    return None, None  # Placeholder until I figure out what the matlab version of this function does


class AutoEncoder:
    # loosely based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
    def __init__(self, inputSize, hiddenSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.totalEpochs = 0

    def encode(self, data, transpose=True):
        if transpose:
            return expit(np.dot(data.transpose(), self.encoderWeights))
        else:
            return expit(np.dot(data, self.encoderWeights))

    def decode(self, data, transpose=True):
        if transpose:
            return expit(np.dot(data, self.decoderWeights)).transpose()
        else:
            return expit(np.dot(data, self.decoderWeights))

    def train(self, data, epochs, randomSeed=28):
        if (self.totalEpochs == 0):
            encoderWeights = tf.Variable(tf.random_normal([self.inputSize, self.hiddenSize], seed=randomSeed))
            decoderWeights = tf.Variable(tf.random_normal([self.hiddenSize, self.inputSize], seed=randomSeed))
        else:
            encoderWeights = tf.Variable(self.encoderWeights)
            decoderWeights = tf.Variable(self.decoderWeights)

        encoded = tf.nn.sigmoid(tf.matmul(data, encoderWeights))
        decoded = tf.nn.sigmoid(tf.matmul(encoded, decoderWeights))

        costFunction = tf.reduce_mean(tf.pow(data - decoded, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(costFunction)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in xrange(epochs):
                _, cost = sess.run([optimizer, costFunction])
                print "Epoch: {0}, Cost: {1}".format(i + 1, cost)
            self.encoderWeights = encoderWeights.eval()
            self.decoderWeights = decoderWeights.eval()

        self.totalEpochs += epochs


def encodeSegment(segment, epochs, gaps, randomSeed=28):
    """
    equivalent of encodeSegment2
    :param segment: input data
    :param epochs: number of epochs to train autoencoder
    :param gaps: boolean describing whether there are gaps
    :param randomSeed: random seed used to generate initial weights
    :return: encoded segment, as well as the autoencoder itself
    """

    # segment MUST be of type float32 to work with tensorflow
    segment = segment.astype(np.float32).transpose()

    inputSize = segment.shape[1]
    if (gaps):
        hiddenSize = inputSize / 100
    else:
        hiddenSize = inputSize / 200

    hiddenSize = 20 if hiddenSize < 20 else hiddenSize

    autoEncoder = AutoEncoder(inputSize, hiddenSize)
    autoEncoder.train(segment, epochs, randomSeed=randomSeed)

    encoded = autoEncoder.encode(segment, transpose=False)

    return encoded, autoEncoder


def numWrong(firstMatrix, secondMatrix):
    """
    equivalent of numWrong
    :param firstMatrix: first matrix being compared
    :param secondMatrix: second matrix being compared
    :return: the number of inconsistencies between the two matrices
    """
    if firstMatrix.shape != secondMatrix.shape:
        print "\n!!ERROR!! in numWrong: Matrices being compared have different shape:"
        print "    matrix 1 is shape {0}, and matrix 2 is shape {1}".format(firstMatrix.shape, secondMatrix.shape)
        return None
    incorrect = 0
    for i in xrange(firstMatrix.shape[0]):
        for j in range(firstMatrix.shape[1]):
            if firstMatrix[i, j] != secondMatrix[i, j]:
                incorrect += 1
    return incorrect


def findErrorMatrix(input, reconstructedInput):
    """
    equivalent of findErrorMatrix
    :param input: original matrix
    :param reconstructedInput: matrix, having  been reconstructed after compression
    :return: error matrix containing position and difference of each inconsistency between the two matrices
    """
    numInconsistencies = numWrong(input, reconstructedInput)
    errorMatrix = np.zeros((numInconsistencies, 3))
    errorMatrixOriginal = input - reconstructedInput
    errorCount = 0
    for i in xrange(errorMatrixOriginal.shape[0]):
        for j in xrange(errorMatrixOriginal.shape[1]):
            if errorMatrixOriginal[i,j] != 0:
                errorMatrix[errorCount, :] = [i, j, errorMatrixOriginal[i, j]]
                errorCount += 1

    return errorMatrix

def decodeVCFComplete(encodedVCF, autoEncoders, errorMatrix, leftOverMatrix):
    scaleRatio = 4
    encodeIndex = 0  # the lowest index of a value not yet decoded

    decodedVCF = None  # IGNORE THIS: it is initialization to stop code linter from getting cranky
    for i in xrange(len(autoEncoders)):
        autoEncoder = autoEncoders[i]
        encodedSubSection = encodedVCF[:, encodeIndex: (encodeIndex + autoEncoder.hiddenSize)]
        decodedSubSection = np.around(autoEncoder.decode(encodedSubSection) * scaleRatio)

        if(i == 0):
            decodedVCF = decodedSubSection
        else:
            decodedVCF = np.vstack((decodedVCF, decodedSubSection))

    for i in xrange(errorMatrix.shape[0]):
        row, col, error = errorMatrix[i]
        decodedVCF[row, col] += error
    return decodedVCF, leftOverMatrix

def getBasesData(filename):
    lettersToNumbers = {'A': 1, 'C': 2, 'T': 3, 'G': 4}
    f = open(filename, 'r')
    data = [list(x.strip()) for x in f.readlines()]
    f.close()
    # from https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python:
    data = [item for sublist in data for item in sublist]
    data = [[lettersToNumbers[letter] for letter in data]]
    return(np.asarray(data).transpose())



def encodeVCFComplete(dataInput, leftOverMatrix, segmentLength = 8000):
    scaleRatio = 4
    leftOverMatrix *= scaleRatio
    dataInput /= scaleRatio
    print dataInput.shape
    numSegments = dataInput.shape[0] / segmentLength if dataInput.shape[0] % segmentLength == 0 else dataInput.shape[0] / segmentLength + 1
    autoEncoders = []
    if numSegments == 1:
        epochs = 3000 if dataInput.shape[0] < segmentLength else 100000 / dataInput.shape[0]
        encodedComplete, autoEncoder = encodeSegment(dataInput, epochs, False)
        autoEncoders.append(autoEncoder)
    else:
        encodedComplete = None  # IGNORE THIS: it is initialization to stop code linter from getting cranky
        for i in xrange(numSegments):
            print "Compressing Segment {0}/{1}".format(i + 1, numSegments)
            startIndex = segmentLength * i
            if i + 1 == numSegments: # if this is the final segment
                dataSegment = dataInput[startIndex:, :]
                print "DATASEGMENT: {0}".format(dataSegment.shape)
                encodedSegment, autoEncoder = encodeSegment(dataSegment, 2000 * 10000 / dataSegment.shape[0], False, randomSeed=28 + i)
                autoEncoders.append(autoEncoder)
                encodedComplete = np.hstack((encodedComplete, encodedSegment))
            else:
                dataSegment = dataInput[startIndex:startIndex + 8000, :]
                epochs = 1500
                encodedSegment, autoEncoder = encodeSegment(dataSegment, epochs, False, randomSeed=28 + i)
                autoEncoders.append(autoEncoder)
                if i == 0:
                    encodedComplete = encodedSegment
                else:
                    encodedComplete = np.hstack((encodedComplete, encodedSegment))
    encodedComplete = simplify(encodedComplete)

    errorMatrix = findErrorMatrix(np.around(dataInput * 4), decodeVCFComplete(encodedComplete, autoEncoders, np.asarray([[0, 0, 0]]), leftOverMatrix)[0])
    return encodedComplete, autoEncoders, errorMatrix, leftOverMatrix

enc, aenc, error, leftover = encodeVCFComplete(getBasesData('bases2'), np.asarray([]))
print error.shape

def encodeGapsComplete(dataInput, leftOverMatrix, segmentLength = 4000):
    """
     equivalent of encodeGapsComplete2 (I know this is almost identical to the encodeVCFComplete function, but I make a separate function
        to match the way they did it in matlab)
    """
    originalDataInput = dataInput.copy()
    dataInput[0, 0] = 0
    scaleRatio = 1
    leftOverMatrix *= scaleRatio

    numSegments = dataInput.shape[0] % segmentLength if dataInput.shape[0] % segmentLength == 0 else dataInput.shape[0] / segmentLength + 1
    autoEncoders = []
    encodedComplete = None  # IGNORE THIS: it is initialization to stop code linter from getting cranky

    if numSegments == 1:
        epochs = min(10000, 2000000000 / dataInput.shape[0])
        encodedComplete, autoEncoder = encodeSegment(dataInput, epochs, False)
    else:
        for i in xrange(numSegments):
            print "Compressing Segment {0}/{1}".format(i, numSegments)
            startIndex = segmentLength * i
            if i + 1 == numSegments: # if this is the final segment
                dataSegment = dataInput[startIndex:, :]
                encodedSegment, autoEncoder = encodeSegment(dataSegment, 6000 * 10000 / dataSegment.shape[0], False, randomSeed=28 + i)
                autoEncoders.append(autoEncoder)
                encodedComplete = np.vstack((encodedComplete, encodedSegment))
            else:
                dataSegment = dataInput[startIndex:startIndex + 8000, :]
                epochs = 10000
                encodedSegment, autoEncoder = encodeSegment(dataSegment, epochs, False, randomSeed=28 + i)
                autoEncoders.append(autoEncoder)
                if i == 0:
                    encodedComplete = encodedSegment
                else:
                    encodedComplete = np.vstack((encodedComplete, encodedSegment))

    encodedComplete = simplify(encodedComplete)

    errorMatrix = findErrorMatrix(np.around(dataInput * scaleRatio), decodeVCFComplete(encodedComplete, autoEncoders, np.asarray([0, 0, 0]), leftOverMatrix)[0])
    errorMatrix = np.vstack((errorMatrix, np.asarray([0, 0, originalDataInput[1] * scaleRatio])))
    return encodedComplete, autoEncoders, errorMatrix, leftOverMatrix



