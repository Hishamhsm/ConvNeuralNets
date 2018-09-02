
# Function to import MNIST Digits image data
def import_and_process_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    return mnist

# Function to import and preprocess cifar10 image data
def import_and_process_cifar10():
    from keras.datasets import cifar10
    import keras.utils as keras_utils   
    
    # Load training and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize the input data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Perform One-Hot Encoding on output values
    y_train = keras_utils.to_categorical(y_train)
    y_test = keras_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test