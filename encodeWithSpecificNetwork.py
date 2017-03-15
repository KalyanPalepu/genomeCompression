#  The encoding scheme here trains a specific network to encode the entire data file
from crestedOwlPort import *

# tuneable values
GAPS_COMPRESSION_FACTOR = 100
GAPS_TRAINING_EPOCHS = 10000

BASES_COMPRESSION_FACTOR = 200
BASES_TRAINING_EPOCHS = 2000


def getBasesData(filename, segmentLength=BASES_SEGMENT_LENGTH):
    """
    Reads data file containing bases (2 bases per line)
    :param filename: file containing bases
    :param segmentLength: length of each segment (default 8000)
    :return: two numpy arrays, with the first being the data that fits cleanly into the segment length,
    and the second being the remainder data that gets left over
    """
    lettersToNumbers = {'A': 1.0, 'C': 2.0, 'T': 3.0, 'G': 4.0}
    f = open(filename, 'r')
    lines = [list(x.strip()) for x in f.readlines()]
    f.close()
    data = [[]]
    # One line contains 2 letters, so we divide segmentLength by 2
    leftOverMatrixIndex = len(lines) - (len(lines) % (segmentLength / 2))
    leftOverMatrix = [[]]
    for line in lines[:leftOverMatrixIndex]:
        for letter in line:
            if letter in lettersToNumbers:
                data[0].append(lettersToNumbers[letter])

    for line in lines[leftOverMatrixIndex:]:
        for letter in line:
            if letter in lettersToNumbers:
                leftOverMatrix[0].append(lettersToNumbers[letter])

    return np.asarray(data), np.asarray(leftOverMatrix)


def getGapsData(filename, segmentLength=GAPS_SEGMENT_LENGTH):
    """
    Reads data file containing bases (1 gap per line)
    :param filename: file containing bases
    :param segmentLength: length of each segment (default 4000)
    :return: two numpy arrays, with the first being the data that fits cleanly into the segment length,
    and the second being the remainder data that gets left over
    """
    f = open(filename)
    data = [int(x.strip()) for x in f.readlines()]
    leftOverMatrixIndex = len(data) - (len(data) % (segmentLength))
    return np.asarray([data[:leftOverMatrixIndex]]), np.asarray([data[leftOverMatrixIndex:]])


def decodeData(encodedData, autoEncoder, errors, leftOverMatrix):
    """
    Decodes data encoded by encodeData()
    :param encodedData: encoded data that needs to be decoded
    :param autoEncoder: autoencoder used for encoding
    :param errors: error matrix used to make compression lossless
    :param leftOverMatrix: see encodeData()
    :param segmentLength: length of one segment
    :return: decoded data and the leftover matrix
    """
    encoded = encodedData.encoded
    compressedSegmentSize = autoEncoder.hiddenSize
    if encoded.shape[1] == 0:
        return np.asarray([[]]), leftOverMatrix

    encodeIndex = 0  # the lowest index of a value not yet decoded
    decoded = np.ndarray((1, 0))
    encodeIndex = 0 # lowest index of a value not yet decoded
    for i in xrange(encoded.shape[1] / compressedSegmentSize):
        encodedSubSection = encoded[:, encodeIndex:(encodeIndex + compressedSegmentSize)]
        decodedSubSection = autoEncoder.decode(encodedSubSection)
        decoded = np.hstack((decoded, decodedSubSection))
        encodeIndex += compressedSegmentSize
    decoded = np.around(denormalize(decoded, encodedData.minValue, encodedData.maxValue))
    for i in xrange(len(errors)):
        row, col, error = errors[i]
        decoded[row, col] += errors
    return decoded, leftOverMatrix


def encodeData(dataInput, leftOverMatrix, segmentLength, compressionFactor, trainingEpochs, batchSize):
    """
    Encodes bases/gaps data with an autoencoder
    :param dataInput: input data (number of bases MUST be divisible by segmentLength;use getBasesData())
    :param leftOverMatrix: leftover data that is not compressed because it does not fill an entire segment (not used right now, but maybe later)
    :param segmentLength: length of each segment (default 8000)
    :param compressionFactor: factor by which the data is compressed (default 200)
    :param trainingEpochs: the number of epochs for which the autoencoder trains
    :param batchSize: the size of each batch fed to the autoencoder
    :return: an EncodedData object with the encoded data, the autoencoder used to compress, an error matrix, and the leftover matrix
    """
    if dataInput.shape[1] < segmentLength:
        print "Data is too small to be compressed.".format(dataInput.shape)
        return np.asarray([[]]), [], np.ndarray([0, 3]), leftOverMatrix

    normalizedData = normalize(dataInput)

    # split data into batches
    numBatches = normalizedData.shape[1] / batchSize
    batches = [normalizedData[:, (i * batchSize):((i + 1) * batchSize)] for i in xrange(numBatches)]

    autoEncoder = AutoEncoder(segmentLength,  20 if segmentLength / compressionFactor < 20 else segmentLength / compressionFactor)
    print "Training Autoencoder..."
    autoEncoder.train(batches, trainingEpochs)

    encoded = np.ndarray((1, 0))

    numSegments = normalizedData.shape[1] / segmentLength
    for i in xrange(numSegments):
        print "Compressing Segment {0}/{1}".format(i + 1, numSegments)
        encoded = np.hstack((encoded, autoEncoder.encode(normalizedData[:, (i * segmentLength):((i + 1) * segmentLength)])))


    encoded = EncodedData(simplify(encoded), dataInput.min(), dataInput.max())  # simplify the data, and convert to EncodedData class

    decoded = decodeData(encoded, autoEncoder, [], leftOverMatrix)[0]


    errorMatrix = findErrorMatrix(dataInput, decoded)

    return encoded, autoEncoder, errorMatrix, leftOverMatrix


def encodeBasesData(dataInput, leftOverMatrix):
    """
    Encodes bases data using tuned parameters
    :param dataInput: see encodeData()
    :param leftOverMatrix: see encodeData()
    :return: see encodeData()
    """
    return encodeData(dataInput, leftOverMatrix, BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, BASES_TRAINING_EPOCHS, BASES_SEGMENT_LENGTH)


def encodeGapsData(dataInput, leftOverMatrix):
    """
    Encodes gaps data using tuned parameters
    :param dataInput: see encodeData()
    :param leftOverMatrix: see encodeData()
    :return: see encodeData()
    """
    return encodeData(dataInput, leftOverMatrix, GAPS_SEGMENT_LENGTH, GAPS_COMPRESSION_FACTOR, GAPS_TRAINING_EPOCHS, GAPS_SEGMENT_LENGTH)

if __name__ == "__main__":
    # Only bases data for now
    filename = raw_input("Name of the file that will be compressed: ")
    encodedData, autoEncoder, errorMatrix, leftOverMatrix = encodeBasesData(*getBasesData(filename))
    accuracy = 100 - ((len(errorMatrix) / float(encodedData.encoded.shape[1] * BASES_COMPRESSION_FACTOR)) * 100)
    print "Compressed with {0:.3f} percent accuracy".format(accuracy)
    np.save('encodedData.npy', encodedData.encoded)
    np.save('encoder.npy', autoEncoder.decoderWeights.astype(np.float16))
    np.save('errorMatrix.npy', errorMatrix)
    np.save('leftOverMatrix.npy', leftOverMatrix)