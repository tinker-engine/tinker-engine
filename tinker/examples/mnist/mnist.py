"""
File contains functions to train and evaluate an MNIST classification model.

Adapted from:
  https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

"""
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


def train(num_epochs, model_name):
    """Train a MNIST classification model."""

    # load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print("Number of images in x_train", x_train.shape[0])
    print("Number of images in x_test", x_test.shape[0])

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, epochs=num_epochs)

    model.evaluate(x_test, y_test)

    model.save(model_name)


def evaluate(index, model_name):
    """Evaluate a trained MNIST classification model."""

    # Load image
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Load model
    reconstructed_model = load_model(model_name)

    # Predict
    pred = reconstructed_model.predict(x_test[index].reshape(1, 28, 28, 1))
    print(pred.argmax())

    # Save prediction image
    plt.imsave("test" + str(index) + "_" + str(pred.argmax()) + ".png", x_test[index], cmap="Greys")
