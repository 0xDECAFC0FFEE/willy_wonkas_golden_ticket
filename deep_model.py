from collections import namedtuple
import tensorflow as tf
import copy

def get_layer_weights(sess, dnn_variables): # TODO: remove
    WeightIndex = namedtuple("WeightIndex", ["weight", "layer_index", "variable_index", "sub_variable_index"])

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

class TF_Wrapper:
    def __init__(self):
        pass

    @classmethod
    def new(cls, layer_definitions, epochs=30, prune_iters=20, prune_style="global", global_prune_rate=.2):
        """
            creates a new model

            layer_definitions: a Layer_Definition list (in layer_definition.py) describing the network architecture
            prune_iters: number of times to repeatedly train and prune
            prune_style: the type of pruning to do. could be "global" or "local".
                local pruning = select some percent of each layer to prune separately.
                global pruning = select some percent of all the weights to prune.
            global_prune_rate: the rate at which to prune the network if prune_style="global". does nothing if prune_style = local
        """
        new_model = TF_Wrapper()
        new_model.layer_definitions = layer_definitions
        new_model.epochs = epochs # number of epochs to run each network for
        new_model.prune_iters = prune_iters # number of times to prune model

        new_model.variables_list = []
        new_model.x, new_model.y, new_model.y_true = None, None, None

        new_model.prune_style = prune_style
        new_model.global_prune_rate = global_prune_rate
        assert prune_style in ["global", "local"]

        new_model.set_layer_shapes()
        new_model.set_layer_weights_random()
        new_model.build_blacklists()
        new_model.build_tf_model()

        return new_model

    def copy(self):
        """
            copies a model over while reinitializing the internal tensorflow code
            doesn't change the blacklist or the initial weights
        """

        new_model = copy.copy(self)
        new_model.x, new_model.y, new_model.y_true = None, None, None
        new_model.layer_definitions = None
        new_model = copy.deepcopy(new_model)

        new_model.layer_definitions = []
        for layer_definition in self.layer_definitions:
            new_model.layer_definitions.append(layer_definition.copy())

        new_model.variables_list = []
        new_model.build_tf_model()

        return new_model

    def set_layer_shapes(self):
        prev_layer = self.layer_definitions[0]
        for layer_definition in self.layer_definitions[1:]:
            layer_definition.set_shape(prev_layer)
            prev_layer = layer_definition

    def set_layer_weights_random(self):
        for layer_definition in self.layer_definitions:
            layer_definition.set_weights_random()

    def build_blacklists(self):
        for layer_definition in self.layer_definitions:
            layer_definition.build_blacklist()

    def build_tf_model(self):
        self.y_true = tf.placeholder(tf.float32, [None, self.layer_definitions[-1].output_shape])
        self.x = self.layer_definitions[0].build_tf_model(None)

        prev_layer_output = self.x
        print(type(self.layer_definitions[0]))
        print("expect", self.layer_definitions[0].input_shape, "=> ",self.layer_definitions[0].output_shape)
        print("actual", self.layer_definitions[0].tf_variable.shape)

        for layer_definition in self.layer_definitions[1:]:
            print(type(layer_definition))
            print("expect", layer_definition.input_shape, "=> ",layer_definition.output_shape)
            prev_layer_output = layer_definition.build_tf_model(prev_layer_output)
            print("actual", layer_definition.tf_variable.shape)

        self.y = prev_layer_output

    def get_layers_weight_variables(self): # TODO: remove
        layers_weight_variables = []
        for layer_definition in layer_definitions:
            layers_weight_variables.append(layer_definition.get_layer_weight_variables())
        return layers_weight_variables

    def get_variable_weights_2(self, sess):
        weights = []
        for layer_index, layer in enumerate(self.layer_definitions):
            weights.extend(layer.get_weight_index_tuples(sess, layer_index))

        return weights

        layer_weights = []
        for layer_index, layer_weight_variables in enumerate(layers_weight_variables):
            for variable_index, layer_weight_variable in enumerate(layer_weight_variables):
                variable = sess.run(layer_weight_variable)
                layer_weights.extend(weights_in_variable(variable, layer_index, variable_index))

    def get_layer_weights(sess, self):
        layer_weights = self.get_layer_weights_2(sess)
        assert get_layer_weights(sess, self.get_layers_weight_variables()) == layer_weights
        print("passed")
        return layer_weights

    def num_weights(self, include_blacklist):
        # counts number of weights in the model
        # set include_blacklist to false to only count number of weights that aren't blacklisted
        num_weights = 0
        for layer in self.layer_definitions:
            num_weights += layer.num_weights(include_blacklist)
        return num_weights

    def clone_definition(self):
        # copies the model INCLUDING the current initial values and blacklists but resetting the tf model
        model_clone = copy.deepcopy(self)
        model_clone.build_tf_model()
        return model_clone

    def prune(self, sess):
        if self.prune_style == "global":
            self.prune_global(sess, self.global_prune_rate)
        elif self.prune_style == "local":
            self.prune_by_layer(sess)
        else:
            raise Exception('unrecognized prune style %s' % self.prune_style)

    def prune_by_layer(self, sess):
        "prunes network layer by layer based on the layer's prune rate"

        for layer in self.layer_definitions:
            layer.prune(weight_index_tuples=None, sess=sess)

    def prune_global(self, sess, rate=.2):
        "then all weights will be lumped into a list and smallest weights will by pruned at the global_rate"

        weights_initial = self.num_weights(include_blacklist=False)
        # print("initial weights num %s" % weights_initial)

        weight_index_tuples = []
        for layer_index, layer in enumerate(self.layer_definitions):
            weight_index_tuples.extend(layer.get_weight_index_tuples(sess, layer_index=layer_index))
            # print("layer %s has %s weights" % (layer_index, layer.num_weights()))

        weight_index_tuples.sort(key=lambda x: abs(x[0]))
        # print("total network weights: %s" % len(weight_index_tuples))

        weight_index_tuples = weight_index_tuples[:int(weights_initial * rate)]
        # print("total network weights to be pruned: %s" % len(weight_index_tuples))

        for i, layer in enumerate(self.layer_definitions):
            layer_weights_to_prune = [weight for weight in weight_index_tuples if weight.layer_index == i]
            layer.prune(weight_index_tuples=layer_weights_to_prune, sess=sess)
