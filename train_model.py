from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
from cnn_model import *
from import_and_process_cifar import *
import utils

mnist = import_and_process_mnist()

X = tf.placeholder(tf.float32, [utils.BATCH_SIZE, utils.NUM_INPUTS])
Y = tf.placeholder(tf.float32, [utils.BATCH_SIZE, utils.NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32)

train_batch_counter = tf.Variable(0, trainable = False, dtype = tf.int64)
increment_counter_op = tf.assign(train_batch_counter, train_batch_counter + 1)
counter_reset_op = tf.assign(train_batch_counter, 0)

logits = conv_net(X, utils.WEIGHTS, utils.BIASES, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = utils.LEARNING_RATE)

train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

progress = {}

def next_batch_provider(batch_size):
    
    batch_counter = train_batch_counter.eval()

    batch_data_to_return = train_data[int(batch_counter * batch_size) : int((batch_counter + 1) * batch_size)]
    batch_labels_to_return = train_labels[int(batch_counter * batch_size) : int((batch_counter + 1) * batch_size)]

    return batch_data_to_return, batch_labels_to_return

with tf.Session() as sess:

    sess.run(init)

    for i in range(utils.NUM_EPOCHS):
        
        for step in range(0, utils.NUM_STEPS):
        
            batch_input, batch_output = mnist.train.next_batch(utils.BATCH_SIZE)

            sess.run(train_op , feed_dict = { X : batch_input, Y : batch_output, keep_prob : 0.7})
            
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_input, Y: batch_output, keep_prob: 1.0})
            
            progress[str(i) + "|" + str(step)] = acc

            print("Loss= " + "{:.4f}".format(loss) + 
            ", Training Accuracy= " + "{:.3f}".format(acc) + 
            ", Epoch|step: " + str(i + 1) + "|" + str(step + 1))
            

        print("------------Next Epoch------------")
        
    print("-----------Optimization Finished-------------")

    peak_accuracy = max(progress, key = progress.get)
    print("Training accuracy reached a peak at Epoch|step: " + peak_accuracy + ". Accuracy was " + str(progress[peak_accuracy]))

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:128], Y: mnist.test.labels[:128], keep_prob: 1.0}))
    
    plt.bar(range(len(progress.keys())), progress.values(), 0.1, color = 'g')
    plt.show()