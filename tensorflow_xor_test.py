import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

def xor_dataset():
	dataset_X = np.random.random(size=[40000, 2])*20-10
	dataset_y = np.array([[1, 0] if (x > 0) != (y > 0) else [0, 1] for x, y in dataset_X])
	return dataset_X, dataset_y

x = tf.placeholder(tf.float32, [None, 2])
y_true = tf.placeholder(tf.float32, [None, 2])

def build_2_layer_model():
	def neuron(X):
		W = tf.Variable(tf.random_normal([2, 1]))
		b = tf.Variable(tf.zeros([1]))
		return tf.sigmoid(tf.matmul(X, W) + b)

	def end_neuron(X):
		W = tf.Variable(tf.random_normal([2, 1]))
		b = tf.Variable(tf.zeros([1]))
		return tf.tanh(tf.matmul(X, W) + b)

	n1 = neuron(x)
	n2 = neuron(x)

	n3 = neuron(tf.concat([n1, n2], 1))
	n4 = neuron(tf.concat([n1, n2], 1))

	return end_neuron(tf.concat([n3, n4], 1))

def build_deep_model(num_inputs, layer_sizes, num_outputs):
	# expecting something like [1, 2, 5, 3] etc
	def layer(X, prev_layer_size, layer_size):
		W = tf.Variable(tf.random_normal([prev_layer_size, layer_size]))
		b = tf.Variable(tf.zeros([layer_size]))
		return tf.sigmoid(tf.matmul(X, W) + b)

	def end_layer(X, prev_layer_size, num_outputs):
		W = tf.Variable(tf.random_normal([prev_layer_size, num_outputs]))
		b = tf.Variable(tf.zeros([num_outputs]))
		return tf.tanh(tf.matmul(X, W) + b)

	prev_layer_size = num_inputs
	prev_layer_output = x

	for layer_size in layer_sizes:
		prev_layer_output = layer(prev_layer_output, prev_layer_size, layer_size)
		prev_layer_size = layer_size

	return end_layer(prev_layer_output, prev_layer_size, num_outputs)

dataset_X, dataset_y = xor_dataset()

y = build_deep_model(2, [8, 8, 8, 2], 2)
# y = build_2_layer_model()

# squared difference
cost = tf.reduce_mean(tf.square(y_true-y))

train_step = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	batches = 20
	batch_size = int(len(dataset_X)/batches)

	for epoch in range(300):
		for batch_index in range(batches):
			Xs = dataset_X[batch_index*batch_size: (batch_index+1)*batch_size]
			ys = dataset_y[batch_index*batch_size: (batch_index+1)*batch_size]

			# Train
			feed = { x: Xs, y_true: ys}
			sess.run(train_step, feed_dict=feed)

			print("cost: %f" % sess.run(cost, feed_dict=feed))

	feed = {x: dataset_X, y: dataset_y}

	wrong_coordinates = []
	right_coordinates = []

	acc_top, acc_bot = 0, 0
	for X_true, y_true, y_pred in zip(dataset_X, dataset_y, sess.run(y, feed_dict={x: dataset_X})):
		if (y_true[0] < .5) != (y_pred[0] < .5):
			wrong_coordinates.append((X_true, y_pred))
		else:
			acc_top += 1
			right_coordinates.append((X_true, y_pred))
		acc_bot += 1


	print("y", sess.run(y, feed_dict={x:[[.1, .1]]}))
	print("y", sess.run(y, feed_dict={x:[[-.1, -.1]]}))
	print("y", sess.run(y, feed_dict={x:[[-.1, .1]]}))
	print("y", sess.run(y, feed_dict={x:[[.1, -.1]]}))


print("accuracy", acc_top/float(acc_bot))

plt.plot([i[0][0] for i in wrong_coordinates], [i[0][1] for i in wrong_coordinates], ".", color="red", )
plt.title("wrong")
plt.show()

plt.plot([i[0][0] for i in right_coordinates], [i[0][1] for i in right_coordinates], ".", color="blue")
plt.title("right")
plt.show()
