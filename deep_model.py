from collections import namedtuple
import numpy as np
import tensorflow as tf
from functools import reduce

class LayerType:
    input_1d = "input_1d"
    input_2d = "input_2d"
    flatten = "flatten"
    dropout = "dropout"
    relu = "relu"
    softmax = "softmax"

LayerDefinition = namedtuple("LayerDefinition", ["type", "params", "init"])
LayerDefinition.__new__.__defaults__ = (None, None, None)

def set_layer_definitions_init_values(layer_definitions):
    prev_layer_size = None
    prev_layer_output = None
    num_weights = 0
    for i, layer_definition in enumerate(layer_definitions):
        layer_type, params, init = layer_definition
        if layer_type == LayerType.input_1d:
            prev_layer_size = params["num_inputs"]
        elif layer_type == LayerType.input_2d:
            prev_layer_size = params["image_width"], params["image_height"]
        elif layer_type == LayerType.flatten:
            prev_layer_size = reduce(lambda a, b: a*b, prev_layer_size, 1)
        elif layer_type == LayerType.dropout:
            pass
        elif layer_type in [LayerType.relu, LayerType.softmax]:
            layer_size = params["layer_size"]
            W_shape = [prev_layer_size, layer_size]
            b_shape = [layer_size]
            W = tf.Session().run(tf.contrib.layers.xavier_initializer()(W_shape))
            b = tf.Session().run(tf.contrib.layers.xavier_initializer()(b_shape))
            layer_definitions[i] = layer_definitions[i]._replace(init=(W, b))

            num_weights += prev_layer_size*layer_size
            num_weights += layer_size

            prev_layer_size = layer_size
        else:
            raise Exception("didnt find layer type", layer_type)
    return num_weights

def fc_layer(X, activation_function, init, blacklist):
    blacklist_W, blacklist_b = blacklist
    W, b = init
    W = tf.where(tf.constant(blacklist_W), tf.Variable(initial_value=W), np.zeros(shape=blacklist_W.shape))
    b = tf.where(tf.constant(blacklist_b), tf.Variable(initial_value=b), np.zeros(shape=blacklist_b.shape))
    return activation_function(tf.matmul(X, W) + b), (W, b)

def build_deep_model(layer_definitions, blacklists):
    x = None # will be set in the init layer
    assert(layer_definitions[-1].type == LayerType.softmax)
    y_true = tf.placeholder(tf.float32, [None, layer_definitions[-1].params["layer_size"]])
    prev_layer_size = None
    prev_layer_output = None
    dnn_variables = []
    for layer_definition, blacklist in zip(layer_definitions, blacklists):
        layer_type, params, init = layer_definition
        if layer_type == LayerType.input_1d:
            num_inputs = params["num_inputs"]
            x = tf.placeholder(tf.float32, [None, num_inputs])
            prev_layer_output = x
            prev_layer_size = num_inputs
            variables = []
        elif layer_type == LayerType.input_2d:
            image_width, image_height = params["image_width"], params["image_height"]
            x = tf.placeholder(tf.float32, [None, image_width, image_height])
            prev_layer_output = x
            prev_layer_size = (image_width, image_height)
            variables = []
        elif layer_type == LayerType.flatten:
            prev_layer_size = prev_layer_size
            prev_layer_output = tf.layers.flatten(prev_layer_output)
            prev_layer_size = reduce(lambda a, b: a*b, prev_layer_size, 1)
            variables = []
        elif layer_type == LayerType.dropout:
            rate = params["rate"]
            prev_layer_output = tf.layers.dropout(prev_layer_output, rate)
            variables = []
        elif layer_type == LayerType.relu:
            layer_size = params["layer_size"]
            prev_layer_output, variables = fc_layer(prev_layer_output, tf.nn.relu, init, blacklist)
            prev_layer_size = layer_size
        elif layer_type == LayerType.softmax:
            layer_size = params["layer_size"]
            prev_layer_output, variables = fc_layer(prev_layer_output, tf.nn.softmax, init, blacklist)
            prev_layer_size = layer_size
        else:
            raise Exception("didnt find layer type", layer_type)
        dnn_variables.append(variables)

    return x, prev_layer_output, y_true, dnn_variables

def build_blacklists(layer_definitions):
    prev_layer_size = None
    prev_layer_output = None
    blacklists = []
    for layer_definition in layer_definitions:
        layer_type, params, init = layer_definition
        if layer_type == LayerType.input_1d:
            prev_layer_size = params["num_inputs"]
            blacklists.append([])
        elif layer_type == LayerType.input_2d:
            prev_layer_size = params["image_width"], params["image_height"]
            blacklists.append([])
        elif layer_type == LayerType.flatten:
            prev_layer_size = reduce(lambda a, b: a*b, prev_layer_size, 1)
            blacklists.append([])
        elif layer_type == LayerType.dropout:
            blacklists.append([])
        elif layer_type in [LayerType.relu, LayerType.softmax]:
            layer_size = params["layer_size"]
            W_blacklist = np.full(shape=(prev_layer_size, layer_size), fill_value=True)
            b_blacklist = np.full(shape=(layer_size), fill_value=True)
            blacklists.append((W_blacklist, b_blacklist))
            prev_layer_size = layer_size
        else:
            raise Exception("didnt find layer type", layer_type)

    return blacklists

def get_layer_weights(sess, dnn_variables):
    WeightIndex = namedtuple("WeightIndex", ["weight", "layer", "wb", "index"])

    def get_weights(i_l, i_wb, i_cur, weights):
        if len(weights.shape) > 1:
            weight_indices = []
            for i, weight in enumerate(weights):
                c_weights = get_weights(i_l, i_wb, i_cur+(i,), weight)
                weight_indices.extend(c_weights)
            return weight_indices
        elif len(weights.shape) == 1:
            return [WeightIndex(weight, i_l, i_wb, i_cur+(i,)) for i, weight in enumerate(weights)]
        else:
            raise Exception("unexpected weights value %s" % weighs)

    variables = []
    for i_l, layer in enumerate(dnn_variables):
        for i_wb, weight_variables in enumerate(layer):
            weight_values = sess.run(weight_variables)
            variables.extend(get_weights(i_l, i_wb, tuple(), weight_values))

    return variables

def count_nonzero_weights(sess, dnn_variables):
    def get_weights(weights, cur_count):
        if len(weights.shape) > 1:
            for weight in weights:
                get_weights(weight, cur_count)
        elif len(weights.shape) == 1:
            cur_count[0] += sum([weight != 0 for weight in weights])
        else:
            raise Exception("unexpected weights value %s" % weighs)

    count = [0] # using a list as a pointer
    for layer in dnn_variables:
        for weight_variables in layer:
            weight_values = sess.run(weight_variables)
            get_weights(weight_values, count)

    return count[0]

# def count_nonzero_weights(sess, dnn_variables):
