import tensorflow as tf
import numpy as np
from functools import reduce
from collections import namedtuple
import copy
from pprint import pprint

WeightIndex = namedtuple("WeightIndex", ["weight", "layer_index", "variable_index", "sub_variable_index"])

class LayerDefinition:
    def __init__(self):
        pass

    def set_weights_random(self):
        pass

    def build_blacklist(self):
        self.blacklist = []

    def get_layer_weight_variables(self):
        return []

    def num_weights(self, include_blacklist=True):
        return 0

    def get_weight_index_tuples(self, sess, layer_index):
        return []

    def prune(self, weight_index_tuples=None, sess=None):
        pass

class Input(LayerDefinition):
    def __init__(self, shape):
        LayerDefinition.__init__(self)
        self.input_shape = None
        self.output_shape = shape

    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable = None
        new_model = copy.deepcopy(new_model)

        return new_model

    def set_shape(self, prev_layer):
        raise Exception("Input layer shape must be set in initialization")

    def build_tf_model(self, prev_layer):
        self.tf_variable = tf.placeholder(tf.float32, [None]+self.output_shape)
        return self.tf_variable

class Flatten(LayerDefinition):
    def __init__(self):
        LayerDefinition.__init__(self)

    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable = None
        new_model = copy.deepcopy(new_model)

        return new_model

    def set_shape(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        self.output_shape = reduce(lambda a, b: a*b, prev_layer.output_shape, 1)

    def build_tf_model(self, prev_layer):
        self.tf_variable = tf.layers.flatten(prev_layer)
        return self.tf_variable


class Dropout(LayerDefinition):
    def __init__(self, rate):
        LayerDefinition.__init__(self)
        self.rate = rate

    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable = None
        new_model = copy.deepcopy(new_model)

    def set_shape(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        self.output_shape = prev_layer.output_shape

    def build_tf_model(self, prev_layer):
        self.tf_variable = tf.layers.dropout(prev_layer, rate)
        return self.tf_variable


class Conv(LayerDefinition):
    def __init__(self, kernel_width=3, num_kernels=5):
        self.kernel_width = kernel_width
        self.num_kernels = num_kernels

        self.W_init = None
        self.B_init = None

    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable, new_model.W, new_model.b = None, None, None
        new_model = copy.deepcopy(new_model)

        return new_model

    def set_shape(self, prev_layer):
        assert len(prev_layer.output_shape) in [2, 3] # only accepts images

        self.input_shape = prev_layer.output_shape
        if len(self.input_shape) == 2:
            self.input_shape.append(1) # adding depth to shape

        prev_height, prev_width, prev_depth = self.input_shape

        # width = prev_width - (self.kernel_width-1)
        # height = prev_height - (self.kernel_width-1)

        # self.input_shape = prev_layer.output_shape
        # self.output_shape = [width, height, self.num_kernels]



        width = prev_width
        height = prev_height

        self.input_shape = prev_layer.output_shape
        self.output_shape = [width, height, self.num_kernels]




        # self.W_shape = [self.num_kernels, self.kernel_width, self.kernel_width, prev_depth]
        self.W_shape = [self.kernel_width, self.kernel_width, prev_depth, self.num_kernels]
        self.b_shape = [self.num_kernels]

        assert prev_width >= self.kernel_width
        assert prev_height >= self.kernel_width

    def set_weights_random(self):
        assert self.W_init == None
        assert self.B_init == None

        with tf.Session() as sess:
            W = sess.run(tf.contrib.layers.xavier_initializer()(self.W_shape))
            b = sess.run(tf.contrib.layers.xavier_initializer()(self.b_shape))

        self.W_init = W
        self.b_init = b

    def build_tf_model(self, prev_layer):
        # self.W = [[[[tf.Variable(initial_value=dimension) for dimension in column] for column in row] for row in kernel] for kernel in self.W_init]
        # self.b = [tf.Variable(initial_value=kernel) for kernel in self.b_init]

        # prev_height, prev_width, prev_depth = self.input_shape

        # kernel_sum = []
        # # pprint(list(zip(self.W, self.b)))
        # for kernel_W, kernel_b in zip(self.W, self.b):
        #     convolve_sum = tf.constant(np.zeros(prev_depth, dtype=np.float32))
        #     for y_index, row_W in enumerate(kernel_W):
        #         for x_index, column_W in enumerate(row_W):
        #             y_stop = prev_height - (self.kernel_width-1) + y_index
        #             x_stop = prev_width - (self.kernel_width-1) + x_index
        #             out = prev_layer[:, y_index:y_stop, x_index:x_stop, :] * column_W
        #             convolve_sum += tf.sum(out)
        #     convolve_sum += kernel_b
        #     kernel_sum.append(convolve_sum)

        # self.tf_variable = tf.concat(kernel_sum, axis=0)




        # self.tf_variable = tf.map_fn()
        # print("conv output", self.tf_variable.shape)
        # print("print expected conv output", self.output_shape)
        # return self.tf_variable

        self.W = tf.Variable(initial_value=self.W_init)
        self.b = tf.Variable(initial_value=self.b_init)
        self.tf_variable = tf.nn.conv2d(prev_layer, filter=self.W, strides=[1, 1, 1, 1], padding="VALID")

        return self.tf_variable

class Pool(LayerDefinition):
    def __init__(self):
        LayerDefinition.__init__(self)
    
    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable = None
        new_model = copy.deepcopy(new_model)

        return new_model

    def set_shape(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        height, width, depth = self.input_shape
        self.output_shape = [int(height/2), int(width/2), depth]

    def build_tf_model(self, prev_layer):
        self.tf_variable = tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        return self.tf_variable


class FCLayer(LayerDefinition):
    def __init__(self, layer_size, W_init=None, b_init=None, prune_p=.2):
        LayerDefinition.__init__(self)
        self.layer_size = layer_size
        self.prune_p = prune_p

        self.W_init = W_init
        self.b_init = b_init

    def set_shape(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        self.output_shape = self.layer_size

        self.W_shape = (prev_layer.output_shape, self.layer_size)
        self.b_shape = [self.output_shape]

    def set_weights_random(self):
        assert self.W_init == None
        assert self.b_init == None

        with tf.Session() as sess:
            W = sess.run(tf.contrib.layers.xavier_initializer()(self.W_shape))
            b = sess.run(tf.contrib.layers.xavier_initializer()(self.b_shape))

        self.W_init = W
        self.b_init = b

    def build_blacklist(self):
        self.W_blacklist = np.full(shape=self.W_shape, fill_value=True)
        self.b_blacklist = np.full(shape=self.b_shape, fill_value=True)

    def build_tf_model(self, prev_layer, activation_function):
        self.W = tf.where(tf.constant(self.W_blacklist), tf.Variable(initial_value=self.W_init), np.zeros(shape=self.W_shape))
        self.b = tf.where(tf.constant(self.b_blacklist), tf.Variable(initial_value=self.b_init), np.zeros(shape=self.b_shape))
        self.tf_variable = activation_function(tf.matmul(prev_layer, self.W) + self.b)
        print("fc input shape", self.tf_variable.shape)
        return self.tf_variable

    def get_layer_weight_variables(self):
        return [self.W, self.b]

    def num_weights(self, include_blacklist=True):
        if include_blacklist:
            return reduce(lambda a, b: a*b, self.W_shape, 1) + self.b_shape[0]
        else:
            return np.sum(self.W_blacklist) + np.sum(self.b_blacklist)

    def get_weight_index_tuples(self, sess, layer_index):
        """
            returns a list of tuples containing a weight and its index in a flattened W or B array
            doesn't include weights that are blacklisted
        """
        weights = []

        W, b = sess.run(self.W), sess.run(self.b)

        flat_W = W.flatten()
        flat_W_blacklist = self.W_blacklist.flatten()

        for W_index, (W_weight, W_blacklist) in enumerate(zip(flat_W, flat_W_blacklist)):
            if W_blacklist:
                weights.append(WeightIndex(weight=W_weight, layer_index=layer_index, variable_index=0, sub_variable_index=W_index))

        for b_index, (b_weight, b_blacklist) in enumerate(zip(b, self.b_blacklist)):
            if b_blacklist:
                weights.append(WeightIndex(weight=b_weight, layer_index=layer_index, variable_index=1, sub_variable_index=b_index))

        return weights

    def prune(self, weight_index_tuples=None, sess=None):
        """
            if weights is a list of WeightIndexes, then those indices will be pruned.
            if weights is none, self.prune_p of the weights will be pruned.
        """

        if weight_index_tuples == None:
            assert sess
            weights_initial = self.num_weights(include_blacklist=False)

            weight_index_tuples = self.get_weight_index_tuples(sess=sess, layer_index=0)
            weight_index_tuples.sort(key=lambda x: abs(x[0]))

            weight_index_tuples = weight_index_tuples[:int(weights_initial * self.prune_p)]

        to_blacklist_W = [w.sub_variable_index for w in weight_index_tuples if w.variable_index == 0]
        to_blacklist_b = [w.sub_variable_index for w in weight_index_tuples if w.variable_index == 1]

        W_blacklist_flat = self.W_blacklist.flatten()
        W_blacklist_flat[to_blacklist_W] = False
        self.W_blacklist = W_blacklist_flat.reshape(self.W_shape)
        self.b_blacklist[to_blacklist_b] = False

class ReLu(FCLayer):
    def __init__(self, layer_size, W_init=None, b_init=None, prune_p=.2):
        FCLayer.__init__(self, layer_size, W_init, b_init, prune_p)

    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable, new_model.W, new_model.b = None, None, None
        new_model = copy.deepcopy(new_model)

        return new_model

    def build_tf_model(self, prev_layer):
        return FCLayer.build_tf_model(self, prev_layer, tf.nn.relu)

class Softmax(FCLayer):
    def __init__(self, layer_size, W_init=None, b_init=None, prune_p=.2):
        FCLayer.__init__(self, layer_size, W_init, b_init, prune_p)

    def copy(self):
        new_model = copy.copy(self)
        new_model.tf_variable, new_model.W, new_model.b = None, None, None
        new_model = copy.deepcopy(new_model)

        return new_model

    def build_tf_model(self, prev_layer):
        return FCLayer.build_tf_model(self, prev_layer, tf.nn.softmax)
