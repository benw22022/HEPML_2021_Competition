"""
Main Code Body
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Sets Tensorflow Logging Level
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import glob
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
import tqdm
from models import net, bayes_net
import numpy as np
from collections import Counter

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

tfd = tfp.distributions


def my_mae(y_true, y_pred):

	y_true = tf.cast(y_true, 'float32')
	y_pred = tf.cast(y_pred, 'float32')
	length = tf.cast(len(y_pred), 'float32')
	mae = tf.math.reduce_sum(tf.abs(y_true - y_pred)) / length
	return mae

	# mae = tf.constant(0.0)
	#
	# for i in range(len(y_true)):
	# 	# print(f"truth -- {true_test_energies[i]} -- predicted -- {predicted_test_energies[i]}")
	# 	# if i > 100:
	# 	# 	break
	# 	mae += tf.abs(y_true[i] - y_pred[i])
	#
	# mae /= len(y_true)
	#
	# return mae



# custom callback for multi-gpu model saving
class ParallelModelCheckpoint(ModelCheckpoint):
	def __init__(self, model, path, monitor='val_loss', verbose=1,
				 save_best_only=False, save_weights_only=True):
		self._model = model
		super(ParallelModelCheckpoint, self).__init__(path, monitor, verbose, save_best_only, save_weights_only)

	def set_model(self, model):
		super(ParallelModelCheckpoint, self).set_model(self._model)


def compute_label(img_name):
	energy = -999
	for i in range(0, len(img_name)):
		if img_name[i] == "keV":
			energy = int(img_name[i - 1])
	return energy


def get_true_energies(files):
	energies = []
	for file in files:
		img_name = file.split("\\")[-1]
		img_name = img_name.split("_")
		energy = -999
		compute_label(img_name)
		energies.append(energy)

	return files, energies


def load_imgage(image_file):
	image = tf.keras.preprocessing.image.load_img(image_file, color_mode='grayscale')
	image = tf.keras.preprocessing.image.img_to_array(image)
	image = np.array([image])
	return image


def make_predictions(image_files, model):
	predicted_energies = []
	for image_file in image_files:
		load_imgage(image_file)
		predicted_energies.append(model.predict(image))
	return predicted_energies


def plot_hist(true_energy, predicted_energy):
	fig2, ax = plt.subplots()
	unique_energies = np.unique(true_energy)

	tmp_dict = {"truth": true_energy.ravel(), "predicted": predicted_energy.ravel()}

	eng_df = pd.DataFrame.from_dict(tmp_dict)

	colours = iter(["blue", "orange", "green", "red", "magenta", "cyan", "yellow", "brown"])

	for unique_energy in unique_energies:
		label = f"{unique_energy} KeV"
		arr = eng_df.loc[eng_df['truth'] == unique_energy]['predicted'].to_numpy().ravel()
		ax.hist(arr, histtype='step', color=next(colours, 'black'), label=label)
	ax.set_xlabel("Predicted Energy KeV")
	ax.set_ylabel("Frequency")
	plt.savefig("test_energy_histograms.svg")
	ax.legend()
	plt.show()


def parse_image(filename):
	parts = tf.strings.split(filename, "_")
	label = compute_label(parts)

	image = tf.io.read_file(filename)
	image = tf.image.decode_jpeg(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	# image = tf.image.resize(image, [128, 128])
	return image, label


def make_ds(file_path):
	list_ds = tf.data.Dataset.list_files(file_path)
	images_ds = list_ds.map(parse_image)
	return images_ds


def show(image, label):
	plt.figure()
	plt.imshow(image)
	plt.title(label.numpy())
	plt.axis('off')
	plt.show()

if __name__ == "__main__":
	# Make datasets
	train_images, train_energies = get_true_energies(glob.glob("data\\train\\*.png"))

	images_ds = make_ds("data\\train\\*.png")
	for image, label in images_ds.take(2):
		show(image, label)

	batch_size = 256
	image_size = (576, 576)

	full_dataset = make_ds("data\\train\\*png")

	DATASET_SIZE = len(train_energies)

	train_size = int(0.7 * DATASET_SIZE)
	val_size = int(0.15 * DATASET_SIZE)
	test_size = int(0.15 * DATASET_SIZE)

	full_dataset = full_dataset.shuffle(buffer_size=100)
	train_dataset = full_dataset.take(train_size).batch(batch_size)
	test_dataset = full_dataset.skip(train_size)
	val_dataset = test_dataset.skip(val_size).batch(batch_size)
	test_dataset = test_dataset.take(test_size).batch(batch_size)

	# Initialize model
	model = net()

	# Configure callbacks
	early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=0,
								   restore_best_weights=True)

	model_checkpoint = ParallelModelCheckpoint(model,
											   path=os.path.join("models", 'weights-{epoch:02d}.h5'),
											   monitor="val_loss", save_best_only=True, save_weights_only=True,
											   verbose=0)

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=4e-6)

	callbacks = [early_stopping, model_checkpoint, reduce_lr]

	# train model
	model.layers[-1].bias.assign([np.mean(train_energies)])
	#model.layers[-1].bias.assign([15])
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
	loss = 'MSE' #'mean_squared_logarithmic_error'#my_mae #tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE) # tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
	model.compile(optimizer=optimizer, loss=loss)
	history = model.fit(train_dataset, validation_data=val_dataset, epochs=500, callbacks=callbacks)

	# Plot training history
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='val')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig("loss_history.svg")
	plt.show()

	# Test Model
	predicted_test_energies = model.predict(test_dataset)

	test_loss = model.evaluate(test_dataset)
	print(f"Testing dataset MAE: {np.average(test_loss)}")

	print(f"Testing dataset MAE: {np.average(test_loss)}")


	true_test_energies = np.concatenate([y for x, y in test_dataset], axis=0)

	fig1, ax = plt.subplots()
	ax.scatter(true_test_energies, predicted_test_energies, alpha=0.1, color='orange')
	ax.set_xlabel("True Energy")
	ax.set_ylabel("Test Energy")
	plt.savefig("test_energy_predictions.svg")
	plt.show()

	mae = 0

	for i in range(len(true_test_energies)):
		# print(f"truth -- {true_test_energies[i]} -- predicted -- {predicted_test_energies[i]}")
		# if i > 100:
		# 	break
		mae += abs(true_test_energies[i] - predicted_test_energies[i])

	mae /= len(true_test_energies)

	print(f"Manual MAE  -- {mae}")

	plot_hist(true_test_energies, predicted_test_energies)


	# Fill CSV with predictions
	unseen_dataset = make_ds("data\\test\\*.png").batch(batch_size)
	predicted_energies = model.predict(unseen_dataset)

	files = [os.path.basename(img) for img in glob.glob("data\\test\\*.png")]

	regression_dict = {"id": files, "energy": predicted_energies.ravel()}
	regression_df = pd.DataFrame.from_dict(regression_dict)
	regression_df.to_csv("regression.csv", index=False)
