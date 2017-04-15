import tensorflow as tf
import numpy as np
from timeit import default_timer
import os

# tuneable values
GAPS_SEGMENT_LENGTH = 16000
BASES_SEGMENT_LENGTH = 16000

compressionRatio = 50


def getGapsData(filename):
    """
    Reads data file containing bases (1 gap per line)
    :param filename: file containing bases
    :return: two numpy arrays, with the first being the data that fits cleanly into
        the segment length (size of num_segments x segmentLength),
        and the second being the remainder data that gets left over.
    """
    print "Reading data..."
    data = [[[]]]
    f = open(filename)
    for line in f:
        line = line.strip()
        data[0][0].append([int(line)])
    f.close()

    return np.asarray(data, dtype=np.int8)


def adjustWithErrorMatrix(reconstructedData, errorMatrix):
    losslessData = np.copy(reconstructedData)
    for error in errorMatrix:
        losslessData[error[0]] = error[1]
    return losslessData


def calculateErrorMatrix(originalData, reconstructedData):
    errorMatrix = []
    originalDataPredictions = np.argmax(originalData[0, 0, :, :], axis=1)
    errors = np.argwhere(originalDataPredictions != reconstructedData)
    for error in errors:
        errorMatrix.append([error, originalDataPredictions[error]])
    return errorMatrix


class ConvolutionalAutoEncoder(object):
    def __init__(self, filterWidth, numFilters, filterStride, numChannels=1,
                 encodeConvolveWeights=None, encodeConvolveBiases=None, decodeConvolveWeights=None,
                 decodeConvolveBiases=None, randomSeed=17, learningRate=0.001):
        """
        :param filterWidth: width of the filters used in convolution
        :param numFilters: number of filters used in convolution
        :param filterStride: filter stride of autoencoder
        :param originalDataSize: original width of data
        :param numChannels: depth of the input data (4 if using one hot encoding, 1 if using numbers)
        :param encodeConvolveWeights: optional pre-trained weights (numpy array)
        :param encodeConvolveBiases: optional pre-trained biases (numpy array)
        :param decodeConvolveWeights: optional pre-trained weights (numpy array)
        :param decodeConvolveBiases: optional pre-trained biases (numpy array)
        :param randomSeed:
        """
        self.filterWidth = filterWidth
        self.numFilters = numFilters
        self.filterStride = filterStride
        self.numChannels = numChannels

        if encodeConvolveWeights is None:
            self.encodeConvolveWeights = tf.Variable(
                tf.truncated_normal([1, filterWidth, numChannels, numFilters], seed=randomSeed))
        else:
            self.encodeConvolveWeights = tf.Variable(encodeConvolveWeights)

        if encodeConvolveBiases is None:
            self.encodeConvolveBiases = tf.Variable(tf.truncated_normal([numFilters], seed=randomSeed))
        else:
            self.encodeConvolveBiases = tf.Variable(encodeConvolveBiases)

        if decodeConvolveWeights is None:
            self.decodeConvolveWeights = tf.Variable(
                tf.truncated_normal([1, filterWidth, numChannels, numFilters], seed=randomSeed))
        else:
            self.decodeConvolveWeights = tf.Variable(decodeConvolveWeights)

        if decodeConvolveBiases is None:
            self.decodeConvolveBiases = tf.Variable(tf.truncated_normal([self.numChannels], seed=randomSeed))
        else:
            self.decodeConvolveBiases = tf.Variable(decodeConvolveBiases)

        self.data = tf.placeholder(np.float32, [1, 1, None, self.numChannels])
        self.outputShape = tf.placeholder(np.int32, [4])
        self.encoded = tf.placeholder(np.float32, [1, 1, None, self.numFilters])

        self.encodeConvolve = tf.nn.conv2d(self.data, self.encodeConvolveWeights, [1, 1, self.filterStride, 1], padding='VALID')
        self.encodeConvolve = tf.nn.bias_add(self.encodeConvolve, self.encodeConvolveBiases)
        # self.encodeConvolve = tf.nn.softmax(self.encodeConvolve)
        self.encodeConvolve = tf.nn.relu(self.encodeConvolve)

        self.decodeConvolve = tf.nn.conv2d_transpose(self.encoded, self.decodeConvolveWeights,
                                                     self.outputShape, [1, 1, self.filterStride, 1],
                                                     padding='VALID')
        self.decodeConvolve = tf.nn.bias_add(self.decodeConvolve, self.decodeConvolveBiases)
        # self.decodeConvolve = tf.nn.softmax(self.decodeConvolve)

        self.decodeConvolveTrain = tf.nn.conv2d_transpose(self.encodeConvolve, self.decodeConvolveWeights,
                                                     self.outputShape, [1, 1, self.filterStride, 1],
                                                     padding='VALID')
        self.decodeConvolveTrain = tf.nn.bias_add(self.decodeConvolveTrain, self.decodeConvolveBiases)
        # self.decodeConvolveTrain = tf.nn.softmax(self.decodeConvolveTrain)


        # self.costFunction = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=self.data[0][0], logits=self.decodeConvolveTrain[0][0]))
        self.costFunction = tf.reduce_mean(tf.pow(self.data - self.decodeConvolveTrain, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self.costFunction)
        self.correct_prediction = tf.equal(self.data, tf.round(self.decodeConvolveTrain))
        self.init = tf.global_variables_initializer()


    def encode(self, data, segmentSize, verifyLossless=False):
        """
        Encodes data using trained autoencoder weights
        :param data: data to be encoded
        :param segmentSize: the size of each segment being encoded
        :return: encoded data, in segments of size segmentSize (except for the last segment,
            which may be smaller if the width of the data is not divisible by segmentSize),
            and the error matrix
        """
        segments = []
        for i in xrange(segmentSize, data.shape[2], segmentSize):
            segments.append(data[:, :, (i - segmentSize):i, :])
        segments.append(data[:, :, -(data.shape[2] % segmentSize):, :])
        leftoverSegmentSize = segments[-1].shape[2]
        encodedSegments = []
        segmentsCompleted = 0
        with tf.Session() as sess:
            sess.run(self.init)
            for segment in segments:
                floatSegment = segment.astype(np.float32)
                segmentsCompleted += 1
                print "Encoding Segment {0} out of {1}".format(segmentsCompleted, len(segments))
                encodedSegments.append(sess.run(self.encodeConvolve, feed_dict={self.data: floatSegment}))

        print "Calculating Error Matrix..."
        decodeWithoutErrorMatrix = self.decode(encodedSegments, [], segmentSize, leftoverSegmentSize)
        errorMatrix = calculateErrorMatrix(data, decodeWithoutErrorMatrix)

        if verifyLossless:
            print "Verifying Losslessness..."
            assert np.array_equal(adjustWithErrorMatrix(decodeWithoutErrorMatrix, errorMatrix), np.argmax(data[0, 0, :, :], axis=1))

        return encodedSegments, errorMatrix



    def decode(self, encodedSegments, errorMatrix, outputSegmentSize, leftoverSegmentSize):
        """
        Decodes data using trained autoencoder weights
        :param encodedSegments: array of segments of encoded data
        :param errorMatrix: array of errors from encoding used to make sure the compression process is lossless
        :param outputSegmentSize: size of each segment when decoded
        :param leftoverSegmentSize: size of the final segment in encodedSegment, when decoded
        :return: decoded data
        """

        outputSize = outputSegmentSize * (len(encodedSegments) - 1) + leftoverSegmentSize
        decoded = np.ndarray([outputSize], dtype=np.int8)
        with tf.Session() as sess:
            sess.run(self.init)
            segmentsCompleted = 0
            index = 0
            for segment in encodedSegments[:-1]:
                segmentsCompleted += 1
                print "Decoding Segment {0} out of {1}".format(segmentsCompleted, len(encodedSegments))
                decodedSegment = sess.run(tf.argmax(self.decodeConvolve, 3), feed_dict={self.encoded: segment, self.outputShape: [1, 1, outputSegmentSize, self.numChannels]})
                decoded[index:(index + outputSegmentSize)] = decodedSegment
                index += outputSegmentSize

            print "Decoding Segment {0} out of {0}".format(len(encodedSegments))
            decodedSegment = sess.run(tf.argmax(self.decodeConvolve, 3), feed_dict={self.encoded: encodedSegments[-1], self.outputShape: [1, 1, leftoverSegmentSize, self.numChannels]})
            decoded[index:] = decodedSegment
        adjustWithErrorMatrix(decoded, errorMatrix)

        return decoded


    def train(self, batches, epochs, saveRate=500, accuracyCalculationRate=4):
        """
        Trains autoencoder to fit the data using TensorFlow
        :param batches: Data that will be trained on (split into an array of batches)
        :param epochs: number of epochs to run training
        :param learningRate: learning rate used in the AdamOptimizer
        :param logDirectory: directory where log containing cost is saved
        :param saveRate: The number of epochs to wait before backing up the weights
        :param accuracyCalculationRate: The number of epochs to wait before calculating and printing the accuracy
        :return: nothing
        """
        # Convert weights to TensorFlow variable
        directoryName = "{0}x{1}x{2}".format(self.numFilters, self.filterWidth, self.filterStride)
        if not os.path.exists(directoryName):
            os.makedirs(directoryName + '/weights/final')
        #
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session() as sess:
            sess.run(self.init)
            logfile = open(directoryName + '/log.txt', 'w')
            avgAccuracy = 0
            # batchAccuracy = []
            for batch in batches:
                floatBatch = batch.astype(np.float32)
                accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                accuracy = sess.run(accuracy, feed_dict={self.data: floatBatch, self.outputShape: batch.shape})
                avgAccuracy += accuracy
                # batchAccuracy.append(accuracy / len(batch))
            print "Pre-Train Accuracy: {0}".format(avgAccuracy / len(batches))
            floatBatch = batches[0].astype(np.float32)
            print batches[0].shape
            for epoch in xrange(epochs):
                start = default_timer()
                avgCost = 0
                avgAccuracy = 0
                totalRight = 0
                for batch in batches:
                    floatBatch = batch.astype(np.float32)
                    _, cost = sess.run([self.optimizer, self.costFunction], feed_dict={self.data: floatBatch, self.outputShape: batch.shape})
                    avgCost += cost
                    if epoch % accuracyCalculationRate == 0:
                        avgAccuracy += sess.run(tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)), feed_dict={self.data: floatBatch, self.outputShape: batch.shape})
                        totalRight += sess.run(tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32)), feed_dict={self.data: floatBatch, self.outputShape: batch.shape})
                    if epoch % saveRate == 0:
                        epochDir = directoryName + '/weights/epoch-' + str(epoch)
                        if not os.path.exists(epochDir):
                            os.makedirs(epochDir)
                        np.save(epochDir + '/encodeConvolveWeights', self.encodeConvolveWeights.eval())
                        np.save(epochDir + '/encodeConvolveBiases.npy', self.encodeConvolveBiases.eval())
                        np.save(epochDir + '/decodeConvolveWeights.npy', self.decodeConvolveWeights.eval())
                        np.save(epochDir + '/decodeConvolveBiases.npy', self.decodeConvolveBiases.eval())
                np.save(directoryName + '/encodeConvolveWeights', self.encodeConvolveWeights.eval())
                np.save(directoryName + '/encodeConvolveBiases.npy', self.encodeConvolveBiases.eval())
                np.save(directoryName + '/decodeConvolveWeights.npy', self.decodeConvolveWeights.eval())
                np.save(directoryName + '/decodeConvolveBiases.npy', self.decodeConvolveBiases.eval())
                end = default_timer()
                if epoch % accuracyCalculationRate == 0:
                    print "Epoch: {0}, Cost: {1}, Time: {2}".format(epoch + 1, avgCost / len(batches), end - start)
                    logfile.write("Epoch: {0}, Cost: {1}, Time: {2}".format(epoch + 1, avgCost / len(batches), end - start))
                    accuracy = open(directoryName + '/accuracy', 'w')
                    accuracy.write(str(avgAccuracy / len(batches)))
                    print "Accuracy: {0}%".format(avgAccuracy / len(batches) * 100)
                    print "Total Right: {0}".format(totalRight)

            batchnum = 0
            avgAccuracy = 0
            for batch in batches:
                print "evaluating on batchnum {0}".format(batchnum)
                batchnum += 1
                floatBatch = batch.astype(np.float32)
                # evaluation method from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
                avgAccuracy += sess.run(tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)), feed_dict={self.data: floatBatch, self.outputShape: batch.shape})
            print "Accuracy: {0}".format(avgAccuracy / len(batches))
            accuracy = open(directoryName + '/accuracy', 'w')
            accuracy.write(str(avgAccuracy / len(batches)))


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

if __name__ == "__main__":
    filterWidth = GAPS_SEGMENT_LENGTH
    numFilters = GAPS_SEGMENT_LENGTH / compressionRatio
    filterStride = GAPS_SEGMENT_LENGTH  # making this equal to GAPS_SEGMENT_LENGTH will make this perform exactly the same
                                        # as with the normal autoencoder.  To make the segments overlap, (the goal of the
                                        # convolutional autoencoder), make this less than GAPS_SEGMENT_LENGTH.
                                        # This will decrease the compression ratio by a factor of
                                        # (GAPS_SEGMENT_LENGTH / filterStride).
    numChannels = 4
    randomSeed = 17

    filename = raw_input("Name of the file that will be compressed: ")
    originalData = getGapsData(filename)
    ac = ConvolutionalAutoEncoder(filterWidth, numFilters, filterStride)
    ac.train([originalData], 10000, accuracyCalculationRate=10)
    # batches = []
    # leftoverIndex = 0
    # for i in xrange(batchSize, originalData.shape[2], batchSize):
    #     batches.append(originalData[:, :, (i - batchSize):i, :])
    #     leftoverIndex = i
    # print originalData[:, :, leftoverIndex:, :].shape
    # print originalData.shape
    # print batchSize
