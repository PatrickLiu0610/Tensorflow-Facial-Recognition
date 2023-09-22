import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = tf.keras.utils.image_dataset_from_directory('Data_Sets')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

data = data.map(lambda x, y: (x / 255.0, y))

training_batch_size = int(len(data) * 0.7)
validation_batch_size = int(len(data) * 0.2) + 1
test_batch_size = int(len(data) * 0.1)

training_data = data.take(training_batch_size)
validation_data = data.skip(training_batch_size).take(validation_batch_size)
test_data = data.skip(training_batch_size + validation_batch_size).take(test_batch_size)

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, (3, 3), 1, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, (3, 3), 1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"]
)

model.fit(training_data, epochs=20, validation_data=validation_data, verbose=2)
model.evaluate(test_data, verbose=2)

model.save(os.path.join('models', 'model.h5'))

