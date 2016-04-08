from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import time
import numpy as np
import sklearn.datasets
import sklearn.cross_validation
import sklearn.metrics
import theano
import theano.tensor as T
import lasagne

# ############################### prepare data ###############################

mnist = sklearn.datasets.fetch_mldata('MNIST original')

X = mnist['data'].astype(np.float32) / 255.0
y = mnist['target'].astype("int32")
X_train, X_valid, y_train, y_valid = sklearn.cross_validation.train_test_split(
    X, y, random_state=42, train_size=50000, test_size=10000)
X_train = X_train.reshape(-1, 1, 28, 28)
X_valid = X_valid.reshape(-1, 1, 28, 28)

# visualize all the X_train data


l_in = lasagne.layers.InputLayer(
    shape=(None, 1, 28, 28),
)

l_hidden1 = lasagne.layers.DenseLayer(
    l_in,
    num_units=64,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform(),
)

l_out = lasagne.layers.DenseLayer(
    l_hidden1,
    num_units=10,
    nonlinearity=lasagne.nonlinearities.softmax,
    W=lasagne.init.GlorotUniform(),
)

# ############################### network loss ###############################

# int32 vector
target_vector = T.ivector('y')


def loss_fn(output):
    return T.mean(lasagne.objectives.categorical_crossentropy(output,
                                                              target_vector))

output = lasagne.layers.get_output(l_out)
loss = loss_fn(output)

# ######################## compiling theano functions ########################

print("Compiling theano functions")

# - takes out all weight tensors from the network, in order to compute
#   how the weights should be updated
all_params = lasagne.layers.get_all_params(l_out)

# - calculate how the parameters should be updated
# - theano keeps a graph of operations, so that gradients w.r.t.
#   the loss can be calculated
updates = lasagne.updates.sgd(
    loss_or_grads=loss,
    params=all_params,
    learning_rate=0.001)

# - create a function that also updates the weights
# - this function takes in 2 arguments: the input batch of images and a
#   target vector (the y's) and returns a list with a single scalar
#   element (the loss)
train_fn = theano.function(inputs=[l_in.input_var, target_vector],
                           outputs=[loss],
                           updates=updates)

# - same interface as previous the previous function, but now the
#   output is a list where the first element is the loss, and the
#   second element is the actual predicted probabilities for the
#   input data
valid_fn = theano.function(inputs=[l_in.input_var, target_vector],
                           outputs=[loss, output])

# ################################# training #################################

print("Starting training...")

num_epochs = 25
batch_size = 600
for epoch_num in range(num_epochs):
    start_time = time.time()
    # iterate over training minibatches and update the weights
    num_batches_train = int(np.ceil(len(X_train) / batch_size))
    train_losses = []
    for batch_num in range(num_batches_train):
        batch_slice = slice(batch_size * batch_num,
                            batch_size * (batch_num + 1))
        X_batch = X_train[batch_slice]
        y_batch = y_train[batch_slice]

        loss, = train_fn(X_batch, y_batch)
        train_losses.append(loss)
    # aggregate training losses for each minibatch into scalar
    train_loss = np.mean(train_losses)

    # calculate validation loss
    num_batches_valid = int(np.ceil(len(X_valid) / batch_size))
    valid_losses = []
    list_of_probabilities_batch = []
    for batch_num in range(num_batches_valid):
        batch_slice = slice(batch_size * batch_num,
                            batch_size * (batch_num + 1))
        X_batch = X_valid[batch_slice]
        y_batch = y_valid[batch_slice]

        loss, probabilities_batch = valid_fn(X_batch, y_batch)
        valid_losses.append(loss)
        list_of_probabilities_batch.append(probabilities_batch)
    valid_loss = np.mean(valid_losses)
    # concatenate probabilities for each batch into a matrix
    probabilities = np.concatenate(list_of_probabilities_batch)
    # calculate classes from the probabilities
    predicted_classes = np.argmax(probabilities, axis=1)
    # calculate accuracy for this epoch
    accuracy = sklearn.metrics.accuracy_score(y_valid, predicted_classes)

    total_time = time.time() - start_time
    print("Epoch: %d, train_loss=%f, valid_loss=%f, valid_accuracy=%f, time=%fs"
          % (epoch_num + 1, train_loss, valid_loss, accuracy, total_time))
