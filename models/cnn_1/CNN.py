import tensorflow as tf
import numpy as np

class SimpleCNN(object):
	"""A simple CNN model implemented using the Tensorflow estimator API"""
	def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir,additional_args={}):
		super(SimpleCNN, self).__init__()
		self.model_input_dim_height = model_input_dim_height
		self.model_input_dim_width = model_input_dim_width
		self.model_input_channels = model_input_channels
		self.n_classes = n_classes
		self.model_dir = model_dir

		#model specific variables
		# Training Parameters

		if("learning_rate" in additional_args):
			self.learning_rate = additional_args["learning_rate"]
		else:
			self.learning_rate = 0.001
			print("using default of "+str(self.learning_rate)+ " for " + "learning_rate")	
		
		if("dropout" in additional_args):
			self.dropout = additional_args["dropout"]
		else:
			self.dropout = 0.25
			print("using default of "+str(self.dropout)+ " for " + "dropout")
		
		self.model = None
		self.InitaliseModel(model_dir=self.model_dir)

		
	### Required Model Functions
	def InitaliseModel(self, model_dir="model_dir"):
		opts = tf.GPUOptions(allow_growth = True)
		conf = tf.ConfigProto(gpu_options=opts)
		trainingConfig = tf.estimator.RunConfig(session_config=conf)
		
		self.model = tf.estimator.Estimator(self.model_fn,model_dir=model_dir,config=trainingConfig)
	

	def TrainModel(self, train_x, train_y, batch_size, num_steps):
		# Define the input function for training
		if(type(train_x) != dict):
			input_dict = {"input":train_x}
		else:
			input_dict = train_x
		
		input_fn = tf.estimator.inputs.numpy_input_fn(
		    x=input_dict, y=train_y,
		    batch_size=batch_size, num_epochs=None, shuffle=True)
		# Train the Model
		self.model.train(input_fn, steps=num_steps)


	def EvaluateModel(self,eval_x,eval_y,batch_size):
		if(type(eval_x) != dict):
			input_dict = {"input":eval_x}
		else:
			input_dict = eval_x

		input_fn = tf.estimator.inputs.numpy_input_fn(
		    x=input_dict, y=eval_y,
		    batch_size=batch_size, num_epochs=None, shuffle=False)
		# Train the Model
		return self.model.evaluate(input_fn, steps=1)


	def Predict(self,predict_x):
		if(type(predict_x) != dict):
			input_dict = {"input":predict_x}
		else:
			input_dict = predict_x

		input_fn = tf.estimator.inputs.numpy_input_fn(
		    x=input_dict, y=None,
		    batch_size=128, num_epochs=1, shuffle=False)
		
		predictions = self.model.predict(input_fn)
		return list(predictions)


	def SaveModel(self,save_dir):
		print("model is automatically saved in "+str(self.model_dir))


	def LoadModel(self,load_dir):
		print("model is automatically loaded from "+str(self.model_dir))


	### Model Specific Functions
	def BuildModel(self, x_dict, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, dropout, reuse, is_training):
		# Define a scope for reusing the variables
		with tf.variable_scope('ConvNet', reuse=reuse):
			#TF Estimator input is a dict, in case of multiple inputs
			x = x_dict['input']

			# Incase of flat input vector reshape to match picture format [Height x Width x Channel]
			# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
			x = tf.reshape(x, shape=[-1, model_input_dim_height, model_input_dim_width, model_input_channels],name="reshaped_input_1")

			# Convolution Layer with 32 filters and a kernel size of 5
			conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu,name="conv_1")
			# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
			max_pool_1 = tf.layers.max_pooling2d(conv1, 2, 2,name="max_pool_1")

			# Convolution Layer with 64 filters and a kernel size of 3
			conv2 = tf.layers.conv2d(max_pool_1, 64, 3, activation=tf.nn.relu,name="conv_2")
			# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
			max_pool_2 = tf.layers.max_pooling2d(conv2, 2, 2,name="max_pool_2")

			# Flatten the data to a 1-D vector for the fully connected layer
			try:
				feature_vector = tf.layers.flatten(max_pool_2,name="feature_vector_1")
			except:
				feature_vector = tf.contrib.layers.flatten(max_pool_2)
			
			# Fully connected layer
			fc1 = tf.layers.dense(feature_vector, 1048,name="fully_connected_1")
			# Apply Dropout (if is_training is False, dropout is not applied)
			fc_dropout_1 = tf.layers.dropout(fc1, rate=dropout, training=is_training,name="dropout_1")

			# Output layer, class prediction
			logits = tf.layers.dense(fc_dropout_1, n_classes,name="logits")

			return logits


   	def model_fn(self, features, labels, mode):
	    # Build the neural network
	    # Because Dropout have different behavior at training and prediction time, we
	    # need to create 2 distinct computation graphs that still share the same weights.
	    if(type(features) != dict):
	    	input_dict = {"input":features}
	    else:
	    	input_dict = features

	    logits_train = self.BuildModel(input_dict, self.model_input_dim_height, self.model_input_dim_height, self.model_input_channels, self.n_classes, self.dropout, reuse=False, is_training=True)
	    logits_test = self.BuildModel(input_dict, self.model_input_dim_height, self.model_input_dim_height, self.model_input_channels, self.n_classes, self.dropout, reuse=True, is_training=False)
	    
	    # Predictions
	    pred_classes = tf.argmax(logits_test, axis=1, name="argmax")
	    pred_probas = tf.nn.softmax(logits_test, name="softmax")
	    
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

	additional_args = {"learning_rate":learning_rate}

	cnn_model = SimpleCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir ="mnist", additional_args = additional_args )

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
	