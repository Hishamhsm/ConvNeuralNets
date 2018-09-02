import tensorflow as tf
import utils


#Define function for creating the model
def conv_net(x, W, b, dropout):
    x = tf.reshape(x, shape = [-1, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH, utils.NUM_CHANNELS])

    # Convolutional Layer 1
    conv1 = conv2d(x, W['c1w'], b['c1b'], strides = 1)
    maxpool1 = maxpool2d(conv1, pool_size = 2)
    print(conv1.get_shape())

    # Convolutional Layer 2
    conv2 = conv2d(maxpool1, W['c2w'], b['c2b'], strides = 1)
    maxpool2 = maxpool2d(conv2, pool_size = 2)
    print(conv2.get_shape())

    # Convolutional Layer 2
    conv3 = conv2d(maxpool2, W['c3w'], b['c3b'], strides = 1)
    maxpool3 = maxpool2d(conv3, pool_size = 2)
    print(conv3.get_shape())

    # Flatten for Fully Connected layer
    conv_out = tf.reshape(maxpool3, [-1 , W['d1w'].get_shape().as_list()[0]])
    print(conv_out.get_shape())    

    # Fully Connected Layer
    dense_out1 = dense(conv_out, W['d1w'], b['d1b'])
    print(dense_out1.get_shape())

    # Apply a dropout layer to prevent overfitting
    dropped_out1 = dropout2D(dense_out1, dropout)
    print(dropped_out1.get_shape())

    # Fully Connected Layer
    dense_out2 = dense(dropped_out1, W['d2w'], b['d2b'])
    print(dense_out2.get_shape())
    
    # Apply a dropout layer to prevent overfitting
    dropped_out2 = dropout2D(dense_out2, dropout)
    print(dropped_out2.get_shape())

    # Prediction
    logits = tf.add(tf.matmul(dropped_out2, W['out']), b['out'])
    return logits

# ---Define Layers as functions for better reuseability---

# Define function for Convolution with relu activation
def conv2d(x, W, b, strides = 2):
    out = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    out = tf.nn.bias_add(out, b)
    out = tf.nn.relu(out)
    out = tf.nn.lrn(out, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
    return out

# Define function for 2D maxpooling
def maxpool2d(x, pool_size = 2):
    out = tf.nn.max_pool(x, ksize= [1, pool_size, pool_size, 1], strides = [1, pool_size, pool_size, 1],  padding = 'SAME')
    return out

# Define function for flattening image data into 1D vectors
def flatten(x, w):
    out = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    return out

# Define function for Fully connected layer with relu activation
def dense(x, W, b):
    out = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(out)

# Define Dropout layer
def dropout2D(x, dropout):
    out = tf.nn.dropout(x, dropout)
    return out
