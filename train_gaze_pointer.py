from pathlib import Path
import argparse

import tensorflow as tf
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from gaze_point_tracker_model import GPTracker
from dataset import DataGenerator
from loss import euclidean_loss


FILE_PATH = Path(__file__).parent.resolve()


BATCH_SIZE = 64
TRAIN_BATCH_SIZE = BATCH_SIZE
VAL_BATCH_SIZE = BATCH_SIZE
EPOCHS = 400
# PATIENCE = 20 # Number of epochs with no improvement after which training will be stopped.
# MOMENTUM = 0.9
# DECAY = 5e-4
SAVE_BEST_ONLY = True
SAVE_WEIGHTS_FREQ = 5

# LEARNING_RATE = 1e-3
LEARNING_RATE = 1e-4


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
	ap.add_argument("-s", "--startepoch", type=int, default=0, help="epoch to restart training at")
	args = vars(ap.parse_args())

	training_data_generator = DataGenerator("train", TRAIN_BATCH_SIZE, head_motion_data_use=True)
	validation_data_generator = DataGenerator("val", VAL_BATCH_SIZE, head_motion_data_use=True)

	if args["model"] is None:
		model = GPTracker()
		model = model.net
		model.compile(
						optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
						loss=euclidean_loss,
						metrics=[euclidean_loss, "accuracy"])
	else:
		model = keras.models.load_model(args["model"], custom_objects={'euclidean_loss': euclidean_loss})
		K.set_value(model.optimizer.lr, LEARNING_RATE)

	# used for saving logs and and weights files to givn locations
	callbacks = [
				TensorBoard(log_dir='./logs'), 
				ModelCheckpoint(
								"gaze_pointer_checkpoints/weights-LR" + 
								str(LEARNING_RATE) +
								".E{epoch:05d}-L{loss:.4f}-VL{val_loss:.4f}-VA{val_acc:.3f}.hdf5",
								save_best_only=SAVE_BEST_ONLY, 
								period=SAVE_WEIGHTS_FREQ
								)]
	
	# runs 
	model.fit_generator(
						generator=training_data_generator.generator(),
						validation_data=validation_data_generator.generator(),
						epochs=EPOCHS,
						steps_per_epoch=int(training_data_generator.epoch_steps),
						validation_steps=int(validation_data_generator.epoch_steps),
						verbose=1,
						initial_epoch=args["startepoch"],
						callbacks=callbacks)


if __name__ == "__main__":
	main()
