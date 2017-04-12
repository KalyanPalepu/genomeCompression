# THIS FILE CONTAINS DEPRICATED CODE PORTED FROM THE MATLAB VERSION OF THIS PROJECT
from crestedOwlPort import *

def getBasesData(filename, segmentLength=8000):
    lettersToNumbers = {'A': 1, 'C': 2, 'T': 3, 'G': 4}
    f = open(filename, 'r')
    lines = [list(x.strip()) for x in f.readlines()]
    f.close()

    data = [[]]
    # One line contains 2 letters, so we divide segmentLength by 2
    leftOverMatrixIndex = len(lines) - (len(lines) % (segmentLength / 2))
    leftOverMatrix = [[]]
    for line in lines[:leftOverMatrixIndex]:
        for letter in line:
            data[0].append(lettersToNumbers[letter])
    for line in lines[leftOverMatrixIndex:]:
        for letter in line:
            if letter in lettersToNumbers:
                leftOverMatrix[0].append(lettersToNumbers[letter])


    return np.asarray(data), np.asarray(leftOverMatrix)

def getGapsData(filename, segmentLength=4000):
    f = open(filename)
    data = [int(x.strip()) for x in f.readlines()]
    leftOverMatrixIndex = len(data) - (len(data) % (segmentLength))

    return np.asarray([data[:leftOverMatrixIndex]]), np.asarray([data[leftOverMatrixIndex:]])


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
    segment = segment.astype(np.float32)

    inputSize = segment.shape[1]
    if (gaps):
        hiddenSize = inputSize / 100
    else:
        hiddenSize = inputSize / 200

    hiddenSize = 20 if hiddenSize < 20 else hiddenSize

    autoEncoder = AutoEncoder(inputSize, hiddenSize, randomSeed=randomSeed)
    autoEncoder.train([segment], epochs)

    encoded = autoEncoder.encode(segment)

    return encoded, autoEncoder


def decodeVCFComplete(encodedVCF, autoEncoders, errorMatrix, leftOverMatrix, scaleRatio=1, maxValue=1, minValue=0):
    # sometimes the data is too small to be compressed, and we only have a leftover matrix
    if len(autoEncoders) == 0:
        return np.asarray([[]]), leftOverMatrix

    compressedSegmentSize = autoEncoders[0].hiddenSize
    encodeIndex = 0  # the lowest index of a value not yet decoded
    decodedVCF = np.ndarray((1, 0))  # IGNORE THIS: it is initialization to stop code linter from getting cranky
    for autoEncoder in autoEncoders:
        encodedSubSection = encodedVCF[:, encodeIndex: (encodeIndex + compressedSegmentSize)]
        decodedSubSection = np.around(autoEncoder.decode(encodedSubSection) * scaleRatio)
        decodedVCF = np.hstack((decodedVCF, decodedSubSection))
        encodeIndex += compressedSegmentSize
    # un-normalize
    decodedVCF = (decodedVCF * (maxValue - minValue)) + minValue

    for i in xrange(len(errorMatrix)):
        row, col, error = errorMatrix[i]
        decodedVCF[row, col] += error
    return decodedVCF, leftOverMatrix


def encodeVCFComplete(dataInput, leftOverMatrix, segmentLength=8000, scaleRatio=4):
    if dataInput.shape[1] < segmentLength:
        print "Data is too small to be compressed (Data shape is ({0}).".format(dataInput.shape)
        return np.asarray([[]]), [], np.ndarray([0, 3]), leftOverMatrix
    dataInput /= scaleRatio
    numSegments = dataInput.shape[1] / segmentLength
    autoEncoders = []
    encodedComplete = np.ndarray((1, 0))
    if numSegments == 1:
        epochs = 3000 if dataInput.shape[1] < segmentLength else 100000 / dataInput.shape[1]
        encodedComplete, autoEncoder = encodeSegment(dataInput, epochs, False)
        autoEncoders.append(autoEncoder)
    else:
        for i in xrange(numSegments):
            print "Compressing Segment {0}/{1}".format(i + 1, numSegments)
            startIndex = segmentLength * i
            dataSegment = dataInput[:, startIndex:(startIndex + segmentLength)]
            epochs = 1500
            encodedSegment, autoEncoder = encodeSegment(dataSegment, epochs, False, randomSeed=28 + i)
            autoEncoders.append(autoEncoder)
            if i == 0:
                encodedComplete = encodedSegment
            else:
                encodedComplete = np.hstack((encodedComplete, encodedSegment))
    encodedComplete = simplify(encodedComplete)

    errorMatrix = findErrorMatrix(np.around(dataInput * 4), decodeVCFComplete(encodedComplete, autoEncoders, [[0, 0, 0]], leftOverMatrix, scaleRatio=4)[0])
    return encodedComplete, autoEncoders, errorMatrix, leftOverMatrix


def encodeGapsComplete(dataInput, leftOverMatrix, segmentLength = 4000):
    """
     equivalent of encodeGapsComplete2 (I know this is almost identical to the encodeVCFComplete function, but I make a separate function
        to match the way they did it in matlab)
    """
    if dataInput.shape[1] == 0:
        print "Data is too small to be compressed (Data shape is ({0}).".format(dataInput.shape)
        return np.asarray([[]]), [], np.ndarray([0, 3]), leftOverMatrix

    dataInput = (dataInput - dataInput.min()) / (dataInput.max() - dataInput.min())  # normalize data

    numSegments = dataInput.shape[1] % segmentLength if dataInput.shape[1] % segmentLength == 0 else dataInput.shape[1] / segmentLength + 1
    autoEncoders = []
    encodedComplete = None  # IGNORE THIS: it is initialization to stop code linter from getting cranky

    if numSegments == 1:
        epochs = min(10000, 2000000000 / dataInput.shape[1])
        encodedComplete, autoEncoder = encodeSegment(dataInput, epochs, False)
        autoEncoders.append(autoEncoder)
    else:
        for i in xrange(numSegments):
            print "Compressing Segment {0}/{1}".format(i, numSegments)
            startIndex = segmentLength * i
            dataSegment = dataInput[:, startIndex:(startIndex + segmentLength)]
            epochs = 10000
            encodedSegment, autoEncoder = encodeSegment(dataSegment, epochs, False, randomSeed=28 + i)
            autoEncoders.append(autoEncoder)
            if i == 0:
                encodedComplete = encodedSegment
            else:
                encodedComplete = np.vstack((encodedComplete, encodedSegment))

    encodedComplete = simplify(encodedComplete)
    decoded = decodeVCFComplete(encodedComplete, autoEncoders, np.asarray([[0, 0, 0]]), leftOverMatrix, minValue=dataInput.min(), maxValue=dataInput.max())[0]

    errorMatrix = findErrorMatrix(np.around(dataInput), decoded)
    return encodedComplete, autoEncoders, errorMatrix, leftOverMatrix, dataInput.min(), dataInput.max()
