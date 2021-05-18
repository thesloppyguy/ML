import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np


# image augmentation -> to prevent over fitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'DATA/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')


test_set = test_datagen.flow_from_directory(
    'DATA/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

cnn = tf.keras.models.Sequential()

# convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu', input_shape=[64, 64, 3]))

# pooling layer
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# convolution layer 2
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
        activation='relu'))

# pooling layer 2
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# flattening layer
cnn.add(tf.keras.layers.Flatten())

# full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# out put layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)


test_1 = image.load_img(
    'DATA/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_1 = image.img_to_array(test_1)
test_1 = np.expand_dims(test_1, axis=0)

result = cnn.predict(test_1)

training_set.class_indices
# first index is the result for the pridiction
if result[0][0] == 1:
    print('dog')
else:
    print('cat')
