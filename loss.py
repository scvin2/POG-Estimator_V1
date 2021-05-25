from keras import backend as K


# calculated mean euclidean distance between true and predicted values
def euclidean_loss(y_true, y_pred):
	return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1, keepdims=True)))
