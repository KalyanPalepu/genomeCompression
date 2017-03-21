#  The encoding scheme here trains a specific network to encode the entire data file
from crestedOwlPort import *

# tuneable values
GAPS_COMPRESSION_FACTOR = 60
GAPS_TRAINING_EPOCHS = 100000

BASES_COMPRESSION_FACTOR = 70
BASES_TRAINING_EPOCHS = 50000


def getBasesData(filename):
    """
    Reads data file containing bases (2 bases per line)
    :param filename: file containing bases
    :return: two numpy arrays, with the first being the data that fits cleanly into
        the segment length (size of num_segments x segmentLength),
        and the second being the remainder data that gets left over.
    """
    print "Reading data..."
    lettersToNumbers = {'A': 1.0, 'C': 2.0, 'T': 3.0, 'G': 4.0}
    f = open(filename, 'r')
    lines = [list(x.strip()) for x in f.readlines()]
    f.close()
    convertedData = []
    # One line contains 2 letters, so we divide segmentLength by 2
    leftOverMatrixIndex = len(lines) - (len(lines) % (BASES_SEGMENT_LENGTH / 2)) + 1
    leftOverMatrix = [[]]

    for line in lines:
        for letter in line:
            if letter in lettersToNumbers:
                convertedData.append(lettersToNumbers[letter])
    for line in lines[leftOverMatrixIndex:]:
        for letter in line:
            if letter in lettersToNumbers:
                leftOverMatrix[0].append(lettersToNumbers[letter])

    data = np.ndarray((0, BASES_SEGMENT_LENGTH), dtype=np.float16)
    for segment in xrange(len(convertedData) / BASES_SEGMENT_LENGTH):
        data = np.vstack((data, np.asarray(convertedData[(segment * BASES_SEGMENT_LENGTH):((segment + 1) * BASES_SEGMENT_LENGTH)])))
    return data, np.asarray(leftOverMatrix, dtype=np.int8)


def getGapsData(filename, segmentLength=GAPS_SEGMENT_LENGTH):
    """
    Reads data file containing bases (1 gap per line)
    :param filename: file containing bases
    :param segmentLength: length of each segment
    :return: two numpy arrays, with the first being the data that fits cleanly into
        the segment length (size of num_segments x segmentLength),
        and the second being the remainder data that gets left over.
    """
    print "Reading data..."
    f = open(filename)
    rawData = [int(x.strip()) for x in f.readlines() if int(x.strip()) < 100]
    f.close()
    leftOverMatrixIndex = len(rawData) - (len(rawData) % (segmentLength))
    data = np.ndarray((0, segmentLength), dtype=np.float32)
    for segment in xrange(len(rawData[:leftOverMatrixIndex]) / segmentLength):
        data = np.vstack((data, np.asarray(rawData[(segment * segmentLength):((segment + 1) * segmentLength)])))
    print data.shape[0], data.shape[1]
    return data, np.asarray([rawData[leftOverMatrixIndex:]])


def decodeData(encodedData, autoEncoder, errorMatrix, leftOverMatrix):
    """
    Decodes data encoded by encodeData()
    :param encodedData: encoded data that needs to be decoded
    :param autoEncoder: autoencoder used for encoding
    :param errorMatrix: error matrix used to make compression lossless
    :param leftOverMatrix: see encodeData()
    :return: decoded data and the leftover matrix
    """
    if encodedData.encoded.shape[1] == 0:
        return np.asarray([[]]), leftOverMatrix

    decoded = np.around(denormalize(autoEncoder.decode(encodedData.encoded), encodedData.minValue, encodedData.maxValue))

    for i in xrange(len(errorMatrix)):
        row, col, error = errorMatrix[i]
        decoded[row, col] += error
    return decoded, leftOverMatrix


def encodeData(dataInput, leftOverMatrix, segmentLength, compressionFactor, trainingEpochs, logDirectory='./'):
    """
    Encodes bases/gaps data with an autoencoder
    :param dataInput: input data (number of bases MUST be divisible by segmentLength;use getBasesData())
    :param leftOverMatrix: leftover data that is not compressed because it does not fill an entire segment (not used right now, but maybe later)
    :param segmentLength: length of each segment
    :param compressionFactor: factor by which the data is compressed
    :param trainingEpochs: the number of epochs for which the autoencoder trains
    :param batchSize: the size of each batch fed to the autoencoder
    :param logDirectoryName: see AutoEncoder.train()
    :return: an EncodedData object with the encoded data, the autoencoder used to compress, an error matrix, and the leftover matrix
    """
    if dataInput.shape[0] < 1:
        print "Data is too small to be compressed.".format(dataInput.shape)
        return EncodedData(np.asarray([[]]), 0, 1), [], np.ndarray([0, 3]), leftOverMatrix

    print "Normalizing data..."
    normalizedData = normalize(dataInput)

    autoEncoder = AutoEncoder(segmentLength,  20 if (segmentLength / compressionFactor) < 20 else (segmentLength / compressionFactor))

    print "Training autoencoder..."
    autoEncoder.train([normalizedData], trainingEpochs, logDirectory=logDirectory)

    print "Encoding data..."
    #encoded = EncodedData(simplify(autoEncoder.encode(normalizedData)), dataInput.min(), dataInput.max())
    encoded = EncodedData(autoEncoder.encode(normalizedData), dataInput.min(), dataInput.max())

    print "Generating error matrix..."
    decoded = decodeData(encoded, autoEncoder, [], leftOverMatrix)[0]
    errorMatrix = findErrorMatrix(dataInput, decoded)

    return encoded, autoEncoder, errorMatrix, leftOverMatrix


def encodeBasesData(dataInput, leftOverMatrix, logDirectory='./'):
    """
    Encodes bases data using tuned parameters
    :param dataInput: see encodeData()
    :param leftOverMatrix: see encodeData()
    :param logDirectory: see encodeData()
    :return: see encodeData()
    """
    return encodeData(dataInput, leftOverMatrix, BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, BASES_TRAINING_EPOCHS, logDirectory=logDirectory)


def encodeGapsData(dataInput, leftOverMatrix):
    """
    Encodes gaps data using tuned parameters
    :param dataInput: see encodeData()
    :param leftOverMatrix: see encodeData()
    :return: see encodeData()
    """
    return encodeData(dataInput, leftOverMatrix, GAPS_SEGMENT_LENGTH, GAPS_COMPRESSION_FACTOR, GAPS_TRAINING_EPOCHS)


def gridSearch(segmentLengths, compressionFactors):
    # segmentLengths = [18000, 17000, 16000]
    # compressionFactors = [100, 90, 80]
    g = open('gridSearchResults/gridSearchResults', 'w')
    for segLen in segmentLengths:
        for compFac in compressionFactors:
            print "Trying with {0} for segment length and {1} for compression factor".format(segLen, compFac)
            BASES_SEGMENT_LENGTH = segLen
            BASES_COMPRESSION_FACTOR = compFac
            directoryName = 'gridSearchResults/' + str(BASES_SEGMENT_LENGTH) + 'x' + str(BASES_COMPRESSION_FACTOR) + '/'
            if not os.path.exists(directoryName):
                os.mkdir(directoryName)
            data = getBasesData(filename)
            encodedData, autoEncoder, errorMatrix, leftOverMatrix = encodeBasesData(*data, logDirectory=directoryName)
            decoded = decodeData(encodedData, autoEncoder, errorMatrix, leftOverMatrix)
            assert np.array_equal(data[0], decoded[0])  # make sure compression was lossless
            accuracy = 100 - ((len(errorMatrix) / float(decoded[0].size)) * 100)

            g.write("Compressed with {0} size segments and {1}x compression at {2:.3f} percent accuracy\n".format(BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, accuracy))
            np.save(directoryName + 'encodedData.npy', encodedData.encoded)
            np.save(directoryName + 'decoder.npy', autoEncoder.decoderWeights)
            np.save(directoryName + 'encoder.npy', autoEncoder.encoderWeights)
            np.save(directoryName + 'errorMatrix.npy', np.asarray(errorMatrix, dtype=np.int8))
            np.save(directoryName + 'leftOverMatrix.npy', leftOverMatrix.astype(np.int8))
            f = open(directoryName + 'accuracy', 'w')
            f.write(str(accuracy))
            f.close()

            print "Compressed with {0} size segments and {1}x compression at {2:.3f} percent accuracy".format(BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, accuracy)
    g.close()

if __name__ == "__main__":
    import os
    # Only bases data for now
    filename = raw_input("Name of the file that will be compressed: ")
    #filename = "bases2"
    #filename = "gaps"
    if 'gap' in filename:
        data = getGapsData(filename)
        encodedData, autoEncoder, errorMatrix, leftOverMatrix = encodeGapsData(*data)
    elif 'base' in filename:
        data = getBasesData(filename)
        encodedData, autoEncoder, errorMatrix, leftOverMatrix = encodeBasesData(*data)
    else:
        print "file name must have either base or gap in it"
        os._exit(2)


    decoded = decodeData(encodedData, autoEncoder, errorMatrix, leftOverMatrix)

    assert np.array_equal(data[0], decoded[0])  # make sure compression was lossless

    accuracy = 100 - ((len(errorMatrix) / float(decoded[0].size)) * 100)
    print "Compressed with {0:.3f} percent accuracy".format(accuracy)

    directoryName = filename.split('.')[0] + "Compressed/"
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)

    np.save(directoryName + 'encodedData.npy', encodedData.encoded)
    np.save(directoryName + 'encoder.npy', autoEncoder.encoderWeights)  # not necessary for the final product, but we save it for debugging
    np.save(directoryName + 'decoder.npy', autoEncoder.decoderWeights)
    np.save(directoryName + 'errorMatrix.npy', np.asarray(errorMatrix, dtype=np.int8))
    np.save(directoryName + 'leftOverMatrix.npy', leftOverMatrix.astype(np.int8))

