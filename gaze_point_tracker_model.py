from keras.layers import *
from keras.models import Model
from constants import FACE_DATA_SHAPE, EYE_DATA_SHAPE

FACE_IMG_SHAPE = FACE_DATA_SHAPE
EYE_IMG_SHAPE = EYE_DATA_SHAPE


class GPTracker:
	def __init__(self):
		self.net = self.build_gptracker_network()

	def build_gptracker_network(self):
		inp_face_data_t = Input(shape=FACE_IMG_SHAPE)
		inp_left_eye_t = Input(shape=EYE_IMG_SHAPE)
		inp_right_eye_t = Input(shape=EYE_IMG_SHAPE)

		left_eye_t = self._apply_eye_net(inp_left_eye_t)
		right_eye_t = self._apply_eye_net(inp_right_eye_t)
		face_data_t = self._apply_face_net(inp_face_data_t)

		data_t = self._apply_concat3(left_eye_t, right_eye_t, face_data_t)
		data_t = self._apply_fc(data_t, 128)
		data_t = self._apply_fc(data_t, 64)

		output = self._apply_fc_output(data_t, 2)
		return Model([inp_face_data_t, inp_left_eye_t, inp_right_eye_t], output)

	# flatten layer
	def _apply_flatten(self, x):
		return Flatten()(x)

	# batch normalize layer
	def _apply_batch_normalize(self, x):
		return BatchNormalization()(x)

	# dropout layer
	def _apply_dropout(self, x, rate):
		return Dropout(rate)(x)

	# concat 2 arrays
	def _apply_concat(self, x, y):
		return Concatenate(axis=1)([x, y])

	# concat 3 arrays
	def _apply_concat3(self, x, y, z):
		return Concatenate(axis=1)([x, y, z])

	# output layer
	def _apply_fc_output(self, x, out_dim, actv_func=None):
		return Dense(out_dim, activation=actv_func)(x)

	# fully connected layer
	def _apply_fc(self, x, out_dim):
		fclayer = Dense(out_dim)(x)
		fclayer = BatchNormalization()(fclayer)
		fclayer = Activation("relu")(fclayer)
		return fclayer

	# 2d convolution layer
	def _apply_conv(self, x, f, k=3, s=1, padding='valid'):
		convlayer = Conv2D(f, kernel_size=k, strides=s, padding=padding)(x)
		convlayer = BatchNormalization()(convlayer)
		convlayer = Activation("relu")(convlayer)
		return convlayer

	# 2d pooling layer
	def _apply_pool(self, x, k=2, s=2):
		return MaxPooling2D(pool_size=k, strides=s)(x)

	# Convolutional neural net for facial features
	def _apply_face_net(self, face_data):
		face_net = self._apply_conv(face_data, 64, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_conv(face_net, 64, k=3, s=1, padding='same')
		face_net = self._apply_conv(face_net, 128, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_conv(face_net, 128, k=3, s=1, padding='same')
		face_net = self._apply_conv(face_net, 256, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_conv(face_net, 256, k=3, s=1, padding='same')
		face_net = self._apply_conv(face_net, 512, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_dropout(face_net, 0.3)
		face_net = self._apply_flatten(face_net)
		face_net = self._apply_fc(face_net, 512)
		face_net = self._apply_fc(face_net, 128)
		return face_net

	# convolutional neural net for eye features
	def _apply_eye_net(self, face_data):
		face_net = self._apply_conv(face_data, 64, k=2, s=1, padding='same')
		# face_net = self._apply_conv(face_net, 64, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_conv(face_net, 128, k=2, s=1, padding='same')
		face_net = self._apply_conv(face_net, 128, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_conv(face_net, 256, k=2, s=1, padding='same')
		face_net = self._apply_conv(face_net, 256, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_conv(face_net, 512, k=2, s=1, padding='same')
		face_net = self._apply_conv(face_net, 512, k=3, s=1, padding='same')
		face_net = self._apply_pool(face_net, k=2, s=2)
		face_net = self._apply_dropout(face_net, 0.3)
		face_net = self._apply_flatten(face_net)
		face_net = self._apply_fc(face_net, 512)
		face_net = self._apply_fc(face_net, 128)
		return face_net
