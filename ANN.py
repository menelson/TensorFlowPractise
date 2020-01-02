'''
An example implementation of an ANN (and a DNN) using TensorFlow.
'''

import tensorflow as tf
import numpy as np

# Necessary fix in order to use the v1 tensorflow functionality 
tf.disable_v2_behavior()
#############################################
# Separate NN construction functon goes here
#############################################

# Create a neuron layer: initalised the weights and biases, then constructs the
# layer, and then puts the output through an activation function if one is 
# specified
def neuron_layer(X, n_neurons, name, activation=None):
	# Create a name scope using the name of the layer (to be provided by the user)
	with tf.name_scope(name):
		n_inputs = int(X.get_shape()[1]) # Pick up the inputs from (None, n_inputs)
		stddev = 2 / np.sqrt(n_inputs)

		# Create the relevant shape for the weight matrix, based on connections between inputs and neurons
		init = tf.truncated_normal((n_inputs, n_neurons), stdev=stdev) # Randomize with a Gaussian
		W = tf.Variable(init, name="kernel") # Random weights initiallu
		b = tf.Variable(tf.zeros([n_neurons]), name="bias") # Bias, initialized to zero
		Z = tf.matmul(X, W) + b

		# Option for passing output through an activation function
		if activation is not None:
			return activation(Z)
		else:
			return Z

#############################################
# Shuffle the data in each batch
#############################################

def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) / batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch


#######################################
# Construction phase
#######################################

n_inputs = 28*28  # MNIST		
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# Use placeholder nodes to represent the training data and targets
#reset_default_graph()

# 2D tensor with instances along the first dimension, and features (28*28) along the second
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

# 1D tensor of length determined by the number of instances (undefined at this point)
# Targets correspond to the MNIST numbers being read off in the 28*28 pixel array
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Now create a DNN
# Alternatively, the contrib dense() can be also be used
with tf.name_scope("dnn"):
	he_init = tf.variance_scaling_initializer()
	hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
			activation=tf.nn.relu)
	hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
			activation=tf.nn.relu)
	logits = neuron_layer(hidden2, n_outputs, name="outputs")

# Compute logits before handling the softmax optimization

# Now it's time to compute the cost function. We'll use cross-entropy 
# based on the logits. Then use reduce_mean to compute the mean cross-entropy
# across all instances
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
	loss = tf.reduce_mean(xentropy, name="loss")

# Now we need to to define an optimizer to minimize the cost function.
# It will have a corresponding learning rate
learning_rate = 0.01
with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

# Finally, define a metric for evaluating how effective the loss is
with tf.name_scope("eval"):
	# Checks whether the highest logit corresponds to the target class
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Save the trained model parameters to disk
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#######################################
# Execution phase
#######################################

# First, load the data using keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0 # Set to a numpy array so that one can reshape to a pixel array and rescale the feature values
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Split training sets into separate training and validation sets
X_valid, X_train = X_train[5000:], X_train[:5000]
y_valid, y_train = y_train[5000:], y_train[:5000]

# Now setup the batches and shuffle the data
n_epochs = 40
batch_size = 50

# Load a TensorFlow session to run the training
with tf.Session() as sess:
	init.run() # Init node initializes all of the variables
	for epoch in range(n_epochs): # A single epoch is a complete iteration through the data
	# Iterate through the number of mini-batches at each epoch
		for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) # Run the training operation

# At the end of each epoch, evaluate the model on the last mini-batch and the validation set
acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

# Save the TensorFlow session
save_path = saver.save(sess, "./my_model_final.ckpt")

#######################################
# Use neural net to make predictions
#######################################
with tf.Session() as sess:
	save_path.restore(sess, "./my_model_final.ckpt")
	X_new_scaled = X_test[:20]
	Z = logits.eval(feed_dict={X: X_new_scaled})
	y_pred = np.argmax(Z, axis=1)

	print("Predicted classes:", y_pred)
	print("Actual classes:   ", y_test[:20])

