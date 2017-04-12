#  The encoding scheme here trains a specific network to encode the entire data file
from crestedOwlPort import *

# tuneable values
GAPS_COMPRESSION_FACTOR = 100
GAPS_TRAINING_EPOCHS = 10000

BASES_COMPRESSION_FACTOR = 70
BASES_TRAINING_EPOCHS = 15000


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
        data = np.vstack(
            (data, np.asarray(convertedData[(segment * BASES_SEGMENT_LENGTH):((segment + 1) * BASES_SEGMENT_LENGTH)])))
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
    rawData = [int(x.strip()) for x in f.readlines()]
    f.close()
    leftOverMatrixIndex = len(rawData) - (len(rawData) % (segmentLength))
    data = np.ndarray((0, segmentLength))
    for segment in xrange(len(rawData[:leftOverMatrixIndex]) / segmentLength):
        data = np.vstack((data, np.asarray(data[(segment * segmentLength):((segment + 1) * segmentLength)])))

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

    decoded = np.around(
        denormalize(autoEncoder.decode(encodedData.encoded), encodedData.minValue, encodedData.maxValue))

    for i in xrange(len(errorMatrix)):
        row, col, error = errorMatrix[i]
        decoded[row, col] += error
    return decoded, leftOverMatrix


def encodeData(dataInput, leftOverMatrix, segmentLength, compressionFactor, trainingEpochs, logDirectory='./', regularizationParameter=0):
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

    autoEncoder = AutoEncoder(segmentLength,
                              20 if (segmentLength / compressionFactor) < 20 else (segmentLength / compressionFactor))

    print "Training autoencoder..."
    autoEncoder.train([normalizedData], trainingEpochs, logDirectory=logDirectory, regularizationParameter=regularizationParameter)

    print "Encoding data..."
    encoded = EncodedData(simplify(autoEncoder.encode(normalizedData)), dataInput.min(), dataInput.max())

    print "Generating error matrix..."
    decoded = decodeData(encoded, autoEncoder, [], leftOverMatrix)[0]
    errorMatrix = findErrorMatrix(dataInput, decoded)

    return encoded, autoEncoder, errorMatrix, leftOverMatrix


def encodeBasesData(dataInput, leftOverMatrix, logDirectory='./', regularizationParameter=0):
    """
    Encodes bases data using tuned parameters
    :param dataInput: see encodeData()
    :param leftOverMatrix: see encodeData()
    :param logDirectory: see encodeData()
    :return: see encodeData()
    """
    return encodeData(dataInput, leftOverMatrix, BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, BASES_TRAINING_EPOCHS,
                      logDirectory=logDirectory, regularizationParameter=regularizationParameter)


def encodeGapsData(dataInput, leftOverMatrix):
    """
    Encodes gaps data using tuned parameters
    :param dataInput: see encodeData()
    :param leftOverMatrix: see encodeData()
    :return: see encodeData()
    """
    return encodeData(dataInput, leftOverMatrix, GAPS_SEGMENT_LENGTH, GAPS_COMPRESSION_FACTOR, GAPS_TRAINING_EPOCHS)


def gridSearch(segmentLengths, compressionFactors, regularizationParameters):
    filename = raw_input("Name of the file that will be compressed: ")
    g = open('lowRegParam/gridSearchResults', 'w')
    for regularizationParameter in regularizationParameters:
        for segLen in segmentLengths:
            for compressionFactor in compressionFactors:
                print "Trying with {0} for segment length, {1} for compression factor, and {2} for reg param".format(segLen, compressionFactor, regularizationParameter)
                BASES_SEGMENT_LENGTH = segLen
                BASES_COMPRESSION_FACTOR = compressionFactor
                directoryName = 'lowRegParam/' + str(BASES_SEGMENT_LENGTH) + 'x' + str(BASES_COMPRESSION_FACTOR) + '/'
                if not os.path.exists(directoryName):
                    os.mkdir(directoryName)
                data = getBasesData(filename)
                encodedData, autoEncoder, errorMatrix, leftOverMatrix = encodeBasesData(*data, logDirectory=directoryName, regularizationParameter=regularizationParameter)
                decoded = decodeData(encodedData, autoEncoder, errorMatrix, leftOverMatrix)
                assert np.array_equal(data[0], decoded[0])  # make sure compression was lossless
                trainAccuracy = 100 - ((len(errorMatrix) / float(decoded[0].size)) * 100)

                data, leftOverMatrix = getBasesData('/home/kalyanp/hugeData/bases2.b')
                encoded = EncodedData(simplify(autoEncoder.encode(normalize(data))), data.min(), data.max())

                decoded = decodeData(encoded, autoEncoder, [], leftOverMatrix)[0]
                errorMatrix = findErrorMatrix(data, decoded)

                decoded = decodeData(encoded, autoEncoder, errorMatrix, leftOverMatrix)

                assert np.array_equal(data, decoded[0])  # make sure compression was lossless

                testAccuracy = 100 - ((len(errorMatrix) / float(decoded[0].size)) * 100)

                g.write("Compressed with {0} size segments, {1}x compression and {2} reg param at {3:.3f} percent train accuracy and {4:.3f} test accuracy".format(BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, regularizationParameter, trainAccuracy, testAccuracy))
                np.save(directoryName + 'encodedData.npy', encodedData.encoded)
                np.save(directoryName + 'decoder.npy', autoEncoder.decoderWeights)
                np.save(directoryName + 'encoder.npy', autoEncoder.encoderWeights)
                np.save(directoryName + 'errorMatrix.npy', np.asarray(errorMatrix, dtype=np.int8))
                np.save(directoryName + 'leftOverMatrix.npy', leftOverMatrix.astype(np.int8))

                print "Compressed with {0} size segments, {1}x compression and {2} reg param at {3:.3f} percent train accuracy and {4:.3f} test accuracy".format(BASES_SEGMENT_LENGTH, BASES_COMPRESSION_FACTOR, regularizationParameter, trainAccuracy, testAccuracy)
    g.close()

if __name__ == "__main__":
    import os
    # Only bases data for now
    filename = raw_input("Name of the file that will be compressed: ")
    data = getBasesData(filename)
    encodedData, autoEncoder, errorMatrix, leftOverMatrix = encodeBasesData(*data)
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

    # enc = np.load('lowRegParam/16000x50/encoder.npy')
    # dec = np.load('lowRegParam/16000x50/decoder.npy')
    # autoEncoder = AutoEncoder(enc.shape[0], enc.shape[1])
    # autoEncoder.encoderWeights = enc
    # autoEncoder.decoderWeights = dec
    # data, leftOverMatrix = getBasesData('/home/kalyanp/hugeData/bases2.b')
    #
    # encoded = EncodedData(simplify(autoEncoder.encode(normalize(data))), data.min(), data.max())
    #
    # decoded = decodeData(encoded, autoEncoder, [], leftOverMatrix)[0]
    # errorMatrix = findErrorMatrix(data, decoded)
    #
    # decoded = decodeData(encoded, autoEncoder, errorMatrix, leftOverMatrix)
    #
    # assert np.array_equal(data, decoded[0])  # make sure compression was lossless
    #
    # accuracy = 100 - ((len(errorMatrix) / float(decoded[0].size)) * 100)
    # print "Compressed with {0:.3f} percent accuracy".format(accuracy)



    # gridSearch([16000], [50], [2.0])
