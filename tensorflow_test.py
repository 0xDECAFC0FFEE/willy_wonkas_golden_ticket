import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt
from collections import namedtuple
import time

def batchify(Xs, ys, batches):
    Xs, ys = shuffle(Xs, ys, random_state=0)
    batch_size = int(len(Xs)/batches)
    for i in range(batches):
        yield Xs[batch_size*i: batch_size*(i+1)], ys[batch_size*i: batch_size*(i+1)]

def mnist_dataset():
    raw_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_X, train_y_raw = raw_data[0]
    test_X, test_y_raw = raw_data[1]

    train_X = train_X.reshape((len(train_X), 784))
    test_X = test_X.reshape((len(test_X), 784))

    # normalizing x values in [0, 1]
    train_X = train_X/255.0
    test_X = test_X/255.0

    train_X = np.random.rand(len(train_X), 7)
    test_X = np.random.rand(len(test_X), 7)

    # 1 hot encoding y values
    train_y = np.zeros((len(train_y_raw), 10))
    train_y[np.arange(len(train_y_raw)), train_y_raw] = 1
    test_y = np.zeros((len(test_y_raw), 10))
    test_y[np.arange(len(test_y_raw)), test_y_raw] = 1

    return (train_X, train_y), (test_X, test_y)

def build_model(num_inputs, layer_parameters, layer_def_repository):
    x = tf.placeholder(tf.float32, [None, num_inputs])

    prev_layer_size = num_inputs
    prev_layer_output = x
    variables_list = []

    for layer_desc in layer_parameters:
        layer_type, params, initial_values, blacklist = layer_desc
        if layer_type == "dropout":
            rate = params
            prev_layer_output = tf.layers.dropout(prev_layer_output, rate)
            variables_list.append((layer_type, []))
        elif layer_type in {"dense", "softmax"}:
            layer_size = params
            prev_layer_output, variable = layer_def_repository[layer_type](prev_layer_output, prev_layer_size, layer_size, initial_values, blacklist)
            variables_list.append((layer_type, variable))
        prev_layer_size = layer_size

    last_layer_type, num_outputs, __, __ = layer_parameters[-1]
    y_true = tf.placeholder(tf.float32, [None, num_outputs])
    y = prev_layer_output

    return x, y, y_true, variables_list


def get_initial_weights(layer_parameters):
    # wrote this whole function cause don't want to write xavier initialization wrong.
    # copies a bunch of code from create deep model 
    print(layer_parameters)
    first_layer = layer_parameters.pop(0)
    layer_1_type, num_inputs = first_layer.type, first_layer.param
    assert(layer_1_type == "input")

    def sigmoid_layer(X, prev_layer_size, layer_size, initial_weights, blacklist):
        W = tf.get_variable("W_sigmoid", shape=[prev_layer_size, layer_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_sigmoid", shape=[layer_size], initializer=tf.contrib.layers.xavier_initializer())
        return tf.nn.relu(tf.matmul(X, W) + b), (W, b)

    def softmax_layer(X, prev_layer_size, num_outputs, initial_weights, blacklist):
        W = tf.get_variable("W_softmax", shape=[prev_layer_size, num_outputs], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_softmax", shape=[num_outputs], initializer=tf.contrib.layers.xavier_initializer())
        return tf.nn.softmax(tf.matmul(X, W) + b), (W, b)
    layer_def_repository = {"softmax": softmax_layer, "dense": sigmoid_layer}

    x, y, y_true, variables_list = build_model(num_inputs, layer_parameters, layer_def_repository)

    # initialize weights
    init = tf.global_variables_initializer()

    init_layer_weights_list = []
    blacklist = []
    with tf.Session() as sess:
        sess.run(init)
        for (layer_type, variables) in variables_list:
            if layer_type in {"softmax", "dense"}:
                W_variable, b_variable = variables
                W_variable = sess.run(W_variable)
                b_variable = sess.run(b_variable)
                init_layer_weights_list.append((W_variable, b_variable))
                blacklist.append(np.full(W_variable.shape, True))
            else:
                init_layer_weights_list.append(None)
                blacklist.append(None)
    
    return init_layer_weights_list, blacklist

def build_deep_model_with_init_blacklist(layer_parameters):
    
    first_layer = layer_parameters.pop(0)
    layer_1_type, num_inputs = first_layer.type, first_layer.param
    assert(layer_1_type == "input")

    def sigmoid_layer(X, prev_layer_size, layer_size, init, blacklist):
        W_init, b_init = init
        W = np.empty(shape=[prev_layer_size, layer_size], dtype=tuple)
        b = np.empty(shape=[layer_size], dtype=tuple)

        for layer_index in tqdm(range(layer_size)):
            b[layer_index] = tf.Variable(b_init[layer_index])wa
            for column_index in range(prev_layer_size):
                W[column_index][layer_index] = tf.Variable(W_init[column_index][layer_index])

        for layer_index in tqdm(range(layer_size)):
            bp = b[layer_index]
            for column_index in range(prev_layer_size):
                if blacklist[column_index][layer_index] == False:
                    bp = bp + (W[column_index][layer_index] * X[column_index])
            b[layer_index] = bp

        return tf.nn.relu(tf.stack(list(b)), (W, b))

    def softmax_layer(X, prev_layer_size, layer_size, init, blacklist):
        W_init, b_init = init
        W = np.empty(shape=[prev_layer_size, layer_size], dtype=tf.Variable)
        b = np.empty(shape=[layer_size], dtype=tf.Variable)

        for layer_index in tqdm(range(layer_size)):
            b[layer_index] = tf.Variable(b_init[layer_index])
            for column_index in range(prev_layer_size):
                W[column_index][layer_index] = tf.Variable(W_init[column_index][layer_index])

        for layer_index in tqdm(range(layer_size)):
            bp = b[layer_index]
            for column_index in range(prev_layer_size):
                # if blacklist[column_index][layer_index] != False:
                    bp = bp + W[column_index][layer_index] * X[column_index]
            b[layer_index] = bp
        return tf.nn.softmax(tf.stack(list(b)), (W, b))

    layer_def_repository = {"dense": sigmoid_layer, "softmax": softmax_layer}

    return build_model(num_inputs, layer_parameters, layer_def_repository)
















(train_X, train_y), (test_X, test_y) = mnist_dataset()

MyLayer = namedtuple("MyLayer", ["type", "param", "init", "blacklist"])
MyLayer.__new__.__defaults__ = (None, None, None, None)

model_parameters = [
    MyLayer("input", 7), #MyLayer("input", 784),
    MyLayer("dense", 512), 
    MyLayer("dropout", .2), 
    MyLayer("softmax", 10)
]
# last two inputs to each layer is the blacklist and the initial values

initial_weights, blacklist = get_initial_weights(copy.deepcopy(model_parameters))


for index, (parameter, init_weight, bl) in enumerate(zip(model_parameters[1:], initial_weights, blacklist)):
    l_type, param = parameter.type, parameter.param
    model_parameters[index+1] = MyLayer(l_type, param, init_weight, bl)

x, y, y_true, variables_list = build_deep_model_with_init_blacklist(copy.deepcopy(model_parameters))
compile_start_time = time.time()
print("started compiling", compile_start_time)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))
train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)
init = tf.global_variables_initializer()

compile_end_time = time.time()
print("sess starting", compile_end_time, compile_end_time-compile_start_time)

# training
with tf.Session() as sess:
    sess.run(init)

    print("sess initted")

    acc_over_time = []
    batches = 8
    batch_size = int(len(train_X)/batches)

    for epoch in tqdm(range(10)): #200
        for Xs, ys in batchify(train_X, train_y, batches):
            # Train
            sess.run(train_step, feed_dict={ x: Xs, y_true: ys})

        # check test acc
        pred_ys_max_index = np.argmax(sess.run(y, feed_dict={x: test_X}), axis=1)
        test_ys_max_index = np.argmax(test_y, axis=1)

        test_acc = list(map(lambda x: x[0] == x[1], zip(pred_ys_max_index, test_ys_max_index)))
        test_acc = np.count_nonzero(test_acc)/float(len(test_acc))
        print("test test_acc", test_acc)
        acc_over_time.append(test_acc)

    print(confusion_matrix(test_ys_max_index, pred_ys_max_index))
    plt.plot(acc_over_time)
    plt.show()











# def build_deep_model(model_dimensions, initial_values, blacklist):
#     num_inputs, layer_parameters, num_outputs = model_dimensions
#     assert len(layer_parameters)+1 == len(initial_values)

#     x = tf.placeholder(tf.float32, [None, num_inputs])
#     y_true = tf.placeholder(tf.float32, [None, num_outputs])

#     def sigmoid_layer(X, prev_layer_size, layer_size, initial_value):
#         if initial_value == "xavier":
#             W = tf.get_variable("W", shape=[prev_layer_size, layer_size], initializer=tf.contrib.layers.xavier_initializer())
#             b = tf.get_variable("b", shape=[layer_size], initializer=tf.contrib.layers.xavier_initializer())
#         else:
#             # W = np.a
#             # for layer in range(layer_size):
#             #     for weight in range(prev_layer_size):

#             W = tf.get_variable("W", shape=[prev_layer_size, layer_size], initializer=tf.constant(initial_value[0]))
#             b = tf.get_variable("b", shape=[layer_size], initializer=tf.constant(initial_value[1]))
#         return tf.nn.relu(tf.matmul(X, W) + b), (W, b)


#     def softmax_layer(X, prev_layer_size, num_outputs, initial_value):
#         with tf.name_scope('softmax_node'):
#             if initial_value == "xavier":
#                 W = tf.get_variable("W2", shape=[prev_layer_size, num_outputs], initializer=tf.contrib.layers.xavier_initializer())
#                 b = tf.get_variable("b2", shape=[num_outputs], initializer=tf.contrib.layers.xavier_initializer())
#             else:
#                 W = tf.get_variable("W2", shape=[prev_layer_size, num_outputs], initializer=tf.constant(initial_value[0]))
#                 b = tf.get_variable("b2", shape=[num_outputs], initializer=tf.constant(initial_value[1]))
#             return tf.nn.softmax(tf.matmul(X, W) + b), (W, b)

#     prev_layer_size = num_inputs
#     prev_layer_output = x
#     variables_list = []
#     for layer_desc, initial_value in zip(layer_parameters, initial_values):
#         layer_type, params = layer_desc
#         if layer_type == "dropout":
#             rate = params
#             prev_layer_output = tf.layers.dropout(prev_layer_output, rate)
#             variables_list.append([])
#         elif layer_type == "dense":
#             layer_size = params
#             prev_layer_output, variable = sigmoid_layer(prev_layer_output, prev_layer_size, layer_size, initial_value)
#             prev_layer_size = layer_size
#             variables_list.append(variable)

#     y, variable = softmax_layer(prev_layer_output, prev_layer_size, num_outputs, initial_values[-1])
#     variables_list.append(variable)

#     return x, y, y_true, variables_list












# initial_values = ["xavier", "", "xavier"]
# x, y, y_true, variables_list = build_deep_model(model_dimensions, initial_values)

# # categorical crossentropy
# loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y))

# train_step = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)

# init = tf.global_variables_initializer()

# # tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
# # with tf.Session(tpu_address) as sess:


# blacklist_variables_list = []

# # getting initial variable weights
# initial_weights_list = []
# with tf.Session() as sess:
#     sess.run(init)
#     for variables in variables_list:
#         initial_weights = []
#         for variable in variables:
#             initial_weights.append(sess.run(variable))
        






# for prune_num in range(0):




#     with tf.Session() as sess:
#         acc_over_time = []
#         batches = 8
#         batch_size = int(len(train_X)/batches)

#         for epoch in tqdm(range(10)): #200
#             for Xs, ys in batchify(train_X, train_y, batches):
#                 # Train
#                 sess.run(train_step, feed_dict={ x: Xs, y_true: ys})

#         # check test acc
#             pred_ys_max_index = np.argmax(sess.run(y, feed_dict={x: test_X}), axis=1)
#             test_ys_max_index = np.argmax(test_y, axis=1)

#             test_acc = list(map(lambda x: x[0] == x[1], zip(pred_ys_max_index, test_ys_max_index)))
#             test_acc = np.count_nonzero(test_acc)/float(len(test_acc))
#             print("test test_acc", test_acc)
#             acc_over_time.append(test_acc)

#         print(confusion_matrix(test_ys_max_index, pred_ys_max_index))
#         plt.plot(acc_over_time)
#         plt.show()

#         print("===========final weights=================")
#         for variables in variables_list:
#             for variable in variables:
#                 print(variable)
#                 print(sess.run(variable))






