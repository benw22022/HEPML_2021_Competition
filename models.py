"""
Models
"""

import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions


def bayes_net(num_samples, lr=0.01):
	kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
											  tf.cast(num_samples, dtype=tf.float32))
	model = tf.keras.models.Sequential([
		tfp.layers.Convolution2DFlipout(
			6, kernel_size=5, padding='SAME',
			kernel_divergence_fn=kl_divergence_function,
			activation=tf.nn.relu),
		tf.keras.layers.MaxPooling2D(
			pool_size=[2, 2], strides=[2, 2],
			padding='SAME'),
		tfp.layers.Convolution2DFlipout(
			16, kernel_size=5, padding='SAME',
			kernel_divergence_fn=kl_divergence_function,
			activation=tf.nn.relu),
		tf.keras.layers.MaxPooling2D(
			pool_size=[2, 2], strides=[2, 2],
			padding='SAME'),
		# tfp.layers.Convolution2DFlipout(
		# 	120, kernel_size=5, padding='SAME',
		# 	kernel_divergence_fn=kl_divergence_function,
		# 	activation=tf.nn.relu),
		tf.keras.layers.Flatten(),
		tfp.layers.DenseFlipout(
			84, kernel_divergence_fn=kl_divergence_function,
			activation=tf.nn.relu),
		tfp.layers.DenseFlipout(
			1, kernel_divergence_fn=kl_divergence_function,
			activation="relu")
	])

	# Model compilation.
	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

	# We use the categorical_crossentropy loss since the MNIST dataset contains
	# ten labels. The Keras API will then automatically add the
	# Kullback-Leibler divergence (contained on the individual layers of
	# the model), to the cross entropy loss, effectively
	# calcuating the (negated) Evidence Lower Bound Loss (ELBO)
	model.compile(optimizer, loss='MSE', experimental_run_tf_function=False)

	return model


def net():
	initializer = tf.keras.initializers.HeNormal()
	kernel_regularizer = None # tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)



	data_augmentation = tf.keras.Sequential([
		tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
		tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
		tf.keras.layers.experimental.preprocessing.CenterCrop(120, 120),
		tf.keras.layers.experimental.preprocessing.Rescaling(1. / 100)
	])

	# kernel_regularizer='l1_l2'

	input = tf.keras.layers. Input(shape=(576, 576, 1))
	x = data_augmentation(input)

	x = tf.keras.layers.Conv2D(16, 5, strides=(4, 4), activation='elu', kernel_initializer=initializer, padding="same", kernel_regularizer=kernel_regularizer)(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
	x = tf.keras.layers.Conv2D(32, 3, strides=(1, 1), activation='elu', kernel_initializer=initializer, padding="same", kernel_regularizer=kernel_regularizer)(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

	# x = inception_module(x, 16, 32, 64, 4, 8, 8)
	# x = tf.keras.layers.BatchNormalization()(x)
	# x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
	# x = inception_module(x, 16*2, 32*2, 64*2, 4*2, 8*2, 8*2)

	# x = tf.keras.layers.BatchNormalization()(x)
	# x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
	x = tf.keras.layers.Flatten()(x)
	# x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(3600, activation='elu', kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)(x)
	x = tf.keras.layers.Dense(500, activation='elu', kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)(x)
	x = tf.keras.layers.Dense(500, activation='elu', kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)(x)

	# x = tf.keras.layers.Dense(50, activation='elu', kernel_initializer=initializer)(x)
		#x = tf.keras.layers.Dense(500, activation='relu', kernel_initializer=initializer, kernel_regularizer='l1_l2')(x)
	# x = tf.keras.layers.Dropout(0.5)(x)
	# x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dense(1)(x)

	return tf.keras.Model(inputs=input, outputs=x)


def inception_module(x, filter_1x1_1, filter_1x1_2, filter_3x3_2, filter_1x1_3, filter_5x5_3, filter_maxpool_4):

	# https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/

	initializer = tf.keras.initializers.HeNormal()

	# 1x1 convolution
	conv_1x1_1 = tf.keras.layers.Conv2D(filter_1x1_1, (1, 1), activation='relu', padding='same', kernel_initializer=initializer)(x)

	# 1x1 convolution -> 3x3 convolution
	conv_1x1_2 = tf.keras.layers.Conv2D(filter_1x1_2, (1, 1), activation='relu', padding='same', kernel_initializer=initializer)(x)
	conv_3x3_2 = tf.keras.layers.Conv2D(filter_3x3_2, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv_1x1_2)

	# 1x1 convolution -> 5x5 convolution
	conv_1x1_3 = tf.keras.layers.Conv2D(filter_1x1_3, (1, 1), activation='relu', padding='same', kernel_initializer=initializer)(x)
	conv_5x5_3 = tf.keras.layers.Conv2D(filter_5x5_3, (3, 3), activation='relu', padding='same', kernel_initializer=initializer)(conv_1x1_3)#

	# 3x3 maxpool -> 1x1 convolution
	maxpool_3x3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
	conv_1x1_4 = tf.keras.layers.Conv2D(filter_maxpool_4, (1, 1), activation='relu', padding='same', kernel_initializer=initializer)(maxpool_3x3)

	output = tf.concat([conv_1x1_1, conv_3x3_2, conv_5x5_3, conv_1x1_4], axis=3)

	return output


