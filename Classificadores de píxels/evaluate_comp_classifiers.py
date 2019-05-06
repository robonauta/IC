import sys
import numpy as np
from CNN_Estimator import CNN_Estimator
import os
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import random
from sklearn.model_selection import train_test_split

images_shape = (40, 40, 1)
num_classes = 2

def load_data(x_file, y_file, features='component_only'):
	x = np.load(x_file) 
	if features == 'component_only':
		x = x[:, 40:, :]
	elif features == 'context_only':
		x = x[:, :40, :]

	x /= 255.
	x = x.reshape((-1, 40, 40, 1))

	y = np.load(y_file)
	y_tmp = np.zeros((y.shape[0], 2), np.uint8)
	y_tmp[:, 0] = y
	y_tmp[:, 1] = 1 - y
	y = y_tmp
	# ind = random.sample(range(10000), 100)

	return x, y

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0]) , recall_score(labels[:, 0], predictions[:, 0] > 0.5 , average=None),\
  precision_score(labels[:, 0], predictions[:, 0]  > 0.5 , average=None)


def train_test_split_2(x, y, test_size = 0.2, random_state=0):
    np.random.seed(random_state)
    indices = np.random.permutation(x.shape[0])
    num_train_samples = int(round(x.shape[0] * (1. - test_size)))
    return x[indices[:num_train_samples]], x[indices[num_train_samples:]], y[indices[:num_train_samples]], y[indices[num_train_samples:]]

def build_model_dir(test_part, features):
	return "model_test_file_%s_features_%s" % (os.path.basename(test_part), os.path.basename(features))

def evaluate(features, train_file_x, train_file_y, test_file_x, test_file_y):

	if features != 'both_bayes':
		x, y = load_data(train_file_x, train_file_y, features)

		x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
		del x, y


		learning_rates = [1e-5, 3e-5, 1e-4, 3e-4]
		conv_layers = [([5, 5, 1, 32], [5, 5, 32, 64]), ([3, 3, 1, 32], [3, 3, 32, 64])]
		conv_strides = [[1, 1, 1, 1], [1, 2, 2, 1]]

		# learning_rates = [1e-5, 3e-5]
		# conv_layers = [([5, 5, 1, 32], [5, 5, 32, 64])]
		# conv_strides = [[1, 3, 3, 1]]

		model_dir = build_model_dir(test_file_y, train_file_x)
		cnn_estimator = CNN_Estimator(images_shape, num_classes,
                        learning_rates=learning_rates, conv_layers=conv_layers, conv_strides=conv_strides,
                        model_dir=model_dir, multilabel=False, num_proc=3)

		# cnn_estimator.fit(x_train, y_train, x_val, y_val)
		# del x_train, x_val, y_train, y_val

		# x, y = load_data(test_file_x, test_file_y, features)
		output = cnn_estimator.predict(x_val)
		#print('output:')
		#print(output[:10])

		np.save(model_dir + '/validation_out.npy', output)

		acc, rec, prec = accuracy(output, y_val)
		print(rec)
		print(prec)
		with open('validation_results.txt', 'a') as f:
			f.write('%s\t%s\t%f\t%f\t%f\t%f\t%f\n' % (os.path.basename(test_file_y), train_file_x, acc, 
				rec[0], rec[1], prec[0], prec[1]))

	else:
		model_dir = build_model_dir(test_file_y, 'component_only')
		# cnn_estimator = CNN_Estimator(images_shape, num_classes, 
		# 	model_dir=model_dir, multilabel=False, num_proc=2, num_epochs=2)

		# x, y = load_data(test_file_x, test_file_y, 'component_only')
		# output_1 = cnn_estimator.predict(x)


		# model_dir = build_model_dir(test_file_y, 'context_only')
		# cnn_estimator = CNN_Estimator(images_shape, num_classes, 
		# 	model_dir=model_dir, multilabel=False, num_proc=2, num_epochs=2)

		# x, y = load_data(test_file_x, test_file_y, 'context_only')
		# output_2 = cnn_estimator.predict(x)

		# #print('output 2:')
		# #print(output_2[:10])

		# output = (output_1 + output_2) / 2.

		# #print('output 3:')
		# #print(output[:10])

		# np.save(output, model_dir + '/test_out.npy')

		# acc, rec, prec = accuracy(output, y)
		# print(rec)
		# print(prec)
		# with open('test_results.txt', 'a') as f:
		# 	f.write('%s\t%s\t%f\t%f\t%f\t%f\t%f\n' % (os.path.basename(test_file_y), features, acc, 
		# 		rec[0], rec[1], prec[0], prec[1]))



# test_parts = ['P1', 'P2']
# feature_types = ['component_only', 'context_only', 'both_bayes', 'both_joined']
#for test_part in test_parts:
#	for features in feature_types:
#		evaluate(test_part, features)
#evaluate(test_parts[0], feature_types[1])

if __name__ == "__main__":
   evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
