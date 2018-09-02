import tensorflow as tf


#Constants
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1

NUM_INPUTS = 784
NUM_CLASSES = 10
POOL_SIZE = 2

NUM_STEPS = 10
NUM_EPOCHS = 50

LEARNING_RATE = 0.001
BATCH_SIZE = 128
DROPOUT = 0.7

#Dictionary for Weights for each layer
WEIGHTS = {
    'c1w' : tf.Variable(tf.truncated_normal([3, 3, 1, 32])),
    'c2w' : tf.Variable(tf.truncated_normal([3, 3, 32, 96])),
    'c3w' : tf.Variable(tf.truncated_normal([3, 3, 96, 192])),
    'd1w' : tf.Variable(tf.truncated_normal([4*4*192, 1024])),
    'd2w' : tf.Variable(tf.truncated_normal([1024, 512])),
    'out' : tf.Variable(tf.truncated_normal([512, NUM_CLASSES]))
}

#Dictionary for biases for each layer
BIASES = {
    'c1b' : tf.Variable(tf.truncated_normal([32])),
    'c2b' : tf.Variable(tf.truncated_normal([96])),
    'c3b' : tf.Variable(tf.truncated_normal([192])),
    'd1b' : tf.Variable(tf.truncated_normal([1024])),
    'd2b' : tf.Variable(tf.truncated_normal([512])),
    'out' : tf.Variable(tf.truncated_normal([NUM_CLASSES]))
}
