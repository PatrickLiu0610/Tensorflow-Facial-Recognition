import os
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model(os.path.join('models', 'model.h5'))

for image in os.listdir(os.path.join('Test_Images')):
    img = cv2.imread(os.path.join('Test_Images', image))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    resize = tf.image.resize(img, (256, 256))
    prediction = model.predict(np.expand_dims(resize / 255, 0))
    print(prediction)

    if prediction > 0.5:
        print(f'class_1')
    else:
        print(f'class_2')
