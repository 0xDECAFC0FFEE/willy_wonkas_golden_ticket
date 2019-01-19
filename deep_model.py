from collections import namedtuple
import numpy as np
import tensorflow as tf

LayerParam = namedtuple("ModelLayerParams", ["type", "params", "init"])
LayerParam.__new__.__defaults__ = (None, None, None)

def set_layer_param_init_values(layer_params):
    prev_layer_size = None
    prev_layer_output = None
    for i, layer_desc in enumerate(layer_params):
        layer_type, params, init = layer_desc
        if layer_type == "input":
            prev_layer_size = params
        elif layer_type == "dropout":
            pass
        elif layer_type in ["dense", "output"]:
            layer_size = params
            W_shape = [prev_layer_size, layer_size]
            b_shape = [layer_size]
            W = tf.Session().run(tf.contrib.layers.xavier_initializer()(W_shape))
            b = tf.Session().run(tf.contrib.layers.xavier_initializer()(b_shape))
            layer_params[i] = list(layer_desc)
            assert layer_params[i][2] == None, "second element of layer params should be init values"
            layer_params[i][2] = (W, b)
            layer_params[i] = LayerParam(*layer_params[i])
            prev_layer_size = layer_size
        else:
            raise Exception("didnt find layer type", layer_type)

def relu_layer(X, prev_layer_size, layer_size, init):
    with tf.name_scope('relu_node'):
        W_shape = [prev_layer_size, layer_size]
        b_shape = [layer_size]
        W, b = init
        W = tf.Variable(initial_value=W)
        b = tf.Variable(initial_value=b)
        return tf.nn.relu(tf.matmul(X, W) + b), (W, b)

def softmax_layer(X, prev_layer_size, layer_size, init):
    with tf.name_scope('softmax_node'):
        W_shape = [prev_layer_size, layer_size]
        b_shape = [layer_size]
        W, b = init
        W = tf.Variable(initial_value=W)
        b = tf.Variable(initial_value=b)
        return tf.nn.softmax(tf.matmul(X, W) + b), (W, b)


def build_deep_model(layer_parameters):
    x = None # will be set in the init layer
    y_true = tf.placeholder(tf.float32, [None, layer_parameters[-1].params])

    prev_layer_size = None
    prev_layer_output = None
    dnn_variables = []
    for layer_desc in layer_parameters:
        layer_type, params, init = layer_desc
        if layer_type == "input":
            num_inputs = params
            x = tf.placeholder(tf.float32, [None, num_inputs]) # equivalent to x
            prev_layer_output = x
            prev_layer_size = num_inputs
            variables = []
        elif layer_type == "dropout":
            rate = params
            prev_layer_output = tf.layers.dropout(prev_layer_output, rate)
            variables = []
        elif layer_type == "dense":
            layer_size = params
            prev_layer_output, variables = relu_layer(prev_layer_output, prev_layer_size, layer_size, init)
            prev_layer_size = layer_size
        elif layer_type == "output":
            layer_size = params
            prev_layer_output, variables = softmax_layer(prev_layer_output, prev_layer_size, layer_size, init)
            prev_layer_size = layer_size
        else:
            raise Exception("didnt find layer type", layer_type)
        dnn_variables.append(variables)

    return x, prev_layer_output, y_true, dnn_variables