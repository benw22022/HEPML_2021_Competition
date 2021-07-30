"""
Models
"""

import tensorflow as tf


def net():
	initializer = tf.keras.initializers.HeNormal()
	kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)

	data_augmentation = tf.keras.Sequential([
		# tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
		# tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
		tf.keras.layers.experimental.preprocessing.CenterCrop(120, 120),
		tf.keras.layers.experimental.preprocessing.Rescaling(1. / 100)
	])

	# kernel_regularizer='l1_l2'

	input = tf.keras.layers. Input(shape=(576, 576, 1))
	x = data_augmentation(input)

	x = tf.keras.layers.Conv2D(4, 5, activation='elu', kernel_initializer=initializer, padding="same",
							   kernel_regularizer=kernel_regularizer)(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)


	x = tf.keras.layers.BatchNormalization()(x)
	# x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
	x = tf.keras.layers.Flatten()(x)
	# x = tf.keras.layers.Dropout(0.5)(x)
	# x = tf.keras.layers.Dense(4000, activation='elu', kernel_initializer=initializer,  kernel_regularizer=kernel_regularizer)(x)
	# x = tf.keras.layers.Dropout(0.5)(x)
	# x = tf.keras.layers.Dense(6000, activation='elu', kernel_initializer=initializer, kernel_regularizer=kernel_regularizer )(x)
	# # x = tf.keras.layers.Dense(500, activation='elu', kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)(x)
	# x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dense(100, activation='elu', kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)(x)
		#x = tf.keras.layers.Dense(500, activation='relu', kernel_initializer=initializer, kernel_regularizer='l1_l2')(x)
	# x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Dense(1, activation='relu')(x)

	return tf.keras.Model(inputs=input, outputs=x)




