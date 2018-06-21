import tensorflow as tf
import numpy as np

class SimpleCNN(object):
	"""docstring for SimpleCNN"""
	def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, learning_rate = 0.001, dropout = 0.25, model_dir = "model_dir"):
		super(SimpleCNN, self).__init__()
		self.model_input_dim_height = model_input_dim_height
		self.model_input_dim_width = model_input_dim_width
		self.model_input_channels = model_input_channels
		self.n_classes = n_classes


		#model specific variables
		# Training Parameters
		self.learning_rate = learning_rate
		self.dropout = dropout
		
		self.model = None
		self.InitaliseModel(model_dir=model_dir)

		from tensorflow.python.client import device_lib
		print("detected devices:")
		print(device_lib.list_local_devices())

	
	def InitaliseModel(self, model_dir="model_dir"):
		self.model = tf.estimator.Estimator(self.model_fn,model_dir=model_dir)
	

	def TrainModel(self, train_x, train_y, batch_size, num_steps):
		# Define the input function for training
		input_fn = tf.estimator.inputs.numpy_input_fn(
		    x=train_x, y=train_y,
		    batch_size=batch_size, num_epochs=None, shuffle=True)
		# Train the Model
		self.model.train(input_fn, steps=num_steps)


	def EvaluateModel(self,eval_x,eval_y,batch_size):
		input_fn = tf.estimator.inputs.numpy_input_fn(
		    x=eval_x, y=eval_y,
		    batch_size=batch_size, num_epochs=None, shuffle=False)
		# Train the Model
		return self.model.evaluate(input_fn, steps=1)


	def Predict(self,predict_x):
		input_fn = tf.estimator.inputs.numpy_input_fn(
		    x=predict_x, y=None,
		    batch_size=128, num_epochs=1, shuffle=False)
		
		return list(self.model.predict(input_fn))


	def SaveModel(self,save_dir):
		print("model is automatically saved in 'model_dir'")


	def LoadModel(self,load_dir):
		print("model is automatically loaded from 'model_dir'")


	def BuildModel(self, x, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, dropout, reuse, is_training):
		with tf.device('/gpu:0'):
			# Define a scope for reusing the variables
			with tf.variable_scope('ConvNet', reuse=reuse):
				# TF Estimator input is a dict, in case of multiple inputs
				# x = x_dict['images']


				# Incase of flat input vector reshape to match picture format [Height x Width x Channel]
				# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
				x = tf.reshape(x, shape=[-1, model_input_dim_height, model_input_dim_width, model_input_channels])

				# Convolution Layer with 32 filters and a kernel size of 5
				conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
				# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
				conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

				# Convolution Layer with 64 filters and a kernel size of 3
				conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
				# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
				conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

				# Flatten the data to a 1-D vector for the fully connected layer
				fc1 = tf.contrib.layers.flatten(conv2)

				# Fully connected layer
				fc1 = tf.layers.dense(fc1, 1048)
				# Apply Dropout (if is_training is False, dropout is not applied)
				fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

				# Output layer, class prediction
				logits = tf.layers.dense(fc1, n_classes)

				return logits


   	def model_fn(self, features, labels, mode):
	    # Build the neural network
	    # Because Dropout have different behavior at training and prediction time, we
	    # need to create 2 distinct computation graphs that still share the same weights.
	    logits_train = self.BuildModel(features, self.model_input_dim_height, self.model_input_dim_height, self.model_input_channels, self.n_classes, self.dropout, reuse=False, is_training=True)
	    logits_test = self.BuildModel(features, self.model_input_dim_height, self.model_input_dim_height, self.model_input_channels, self.n_classes, self.dropout, reuse=True, is_training=False)
	    
	    # Predictions
	    pred_classes = tf.argmax(logits_test, axis=1)
	    pred_probas = tf.nn.softmax(logits_test)
	    
	    # If prediction mode, early return
	    if mode == tf.estimator.ModeKeys.PREDICT:
	        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
	        
	    # Define loss and optimizer
	    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
	    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
	    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
	    
	    # Evaluate the accuracy of the model
	    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	    
	    # TF Estimators requires to return a EstimatorSpec, that specify
	    # the different ops for training, evaluating, ...
	    estim_specs = tf.estimator.EstimatorSpec(
	      mode=mode,
	      predictions=pred_classes,
	      loss=loss_op,
	      train_op=train_op,
	      eval_metric_ops={'accuracy': acc_op})

	    return estim_specs


if __name__ == '__main__':
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

	model_input_dim_height = 28
	model_input_dim_width = 28 
	model_input_channels = 1
	n_classes = 10 
	learning_rate = 0.001

	batch_size = 128
	num_train_steps = 200

	cnn_model = SimpleCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, learning_rate = learning_rate)

	verbose_every = 10
	for step in range(verbose_every,num_train_steps+1,verbose_every):
		print("")
		print("training")
		print("step:",step)
		cnn_model.TrainModel(mnist.train.images, mnist.train.labels, batch_size, verbose_every)
		print("")
		
		print("evaluation")
		print(cnn_model.EvaluateModel(mnist.test.images[:128], mnist.test.labels[:128], batch_size))
		print("")
		


	print(cnn_model.Predict(mnist.test.images[:5]))
	

	print(mnist.test.labels[:5])
