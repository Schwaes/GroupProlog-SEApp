import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten, Activation, Dropout, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

X = np.load('feats array.npy')
y = np.load('tags array.npy')

le = LabelEncoder().fit(y)
y_encoded = le.transform(y)
y_encoded.astype('float32').reshape((-1,1))

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y_encoded, test_size = .3)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size = .5)

tag_amounts = {}

for i in y:
	if i not in tag_amounts.keys():
		tag_amounts[i] = 1
	else:
		tag_amounts[i] += 1

unique_tags = tag_amounts.keys()
print(len(unique_tags))

test_scores = []
train_scores = []
val_scores = []
num_labels = len(unique_tags)
'''
for i in [.025, .5, .75]:

	model = Sequential()

	model.add(Conv1D(64, input_shape = (40, )))
	model.add(Activation('relu'))
	model.add(Dropout(.7))

	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(i))

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(i))

	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(i))

	model.add(Dense(256))
	model.add(Activation('softmax'))

	    # Display model architecture summary 
	model.summary()

	model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

	num_epochs = 256
	num_batch_size = 512

	checkpointer = ModelCheckpoint(filepath = 'saved_models/new weights.hdf5', 
	                                   verbose = 1, save_best_only = True)
	start = datetime.now()

	model.fit(X_train, y_train, batch_size = num_batch_size, epochs = num_epochs, validation_data = (X_test, y_test), callbacks = [checkpointer], verbose = 1)

	duration = datetime.now() - start
	print("Training completed in time: ", duration)

	train_accuracy = model.evaluate(X_train, y_train, verbose = 0)
	test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
	test_scores.append(test_accuracy[1])
	train_scores.append(train_accuracy[1])

	val_accuracy = model.evaluate(X_val, y_val, verbose = 0)
	val_scores.append(val_accuracy[1])

print(train_scores)
print(test_scores)
print(val_scores)
'''
#Conv1D Model

print(X_train.shape)

X_train = X_train.reshape(-1,40,1)
X_test = X_test.reshape(-1,40,1)

drop_out_rate = 0.2

input_tensor = Input(shape=(40, 1))

x = layers.Conv1D(32, 39, activation='relu', strides=1)(input_tensor)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
output_tensor = layers.Dense(256, activation='softmax')(x)

model = keras.Model(input_tensor, output_tensor)

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=keras.optimizers.Adam(lr = .003),
             metrics=['accuracy'])

num_epochs = 256
num_batch_size = 512

checkpointer = ModelCheckpoint(filepath = 'saved_models/new weights.hdf5', 
	                                   verbose = 1, save_best_only = True)
start = datetime.now()

model.fit(X_train, y_train, batch_size = num_batch_size, epochs = num_epochs, validation_data = (X_test, y_test), callbacks = [checkpointer], verbose = 1)

duration = datetime.now() - start
print("Training completed in time: ", duration)

train_accuracy = model.evaluate(X_train, y_train, verbose = 0)
test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
test_scores.append(test_accuracy[1])
train_scores.append(train_accuracy[1])

val_accuracy = model.evaluate(X_val, y_val, verbose = 0)
val_scores.append(val_accuracy[1])

print(train_scores)
print(test_scores)
print(val_scores)