from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input, Lambda
import tensorflow as tf        

def fft_layer(x):
    """Applies 2D FFT to each channel independently and returns the magnitude spectrum."""
    x = tf.cast(x, tf.complex64)  # Convert real to complex
    fft = tf.signal.fft2d(x)  # Compute 2D FFT per channel

    fft_magnitude = tf.abs(fft)  # Use magnitude only
    return fft_magnitude

def LPQ_net(in_shape=(256,256,3), num_classes=2):
    model = Sequential()

    model.add(Input(shape=in_shape))

    # Apply FFT as the first layer
    model.add(Lambda(fft_layer, name="fft_layer"))

    model.add(Conv2D(32, kernel_size=(3, 3), name='convRes', activation='relu'))

    model.add(Conv2D(32, kernel_size=(5, 5), name='conv1', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), name='pool1'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), name='conv2', activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(5, 5), name='conv3', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), name='pool2'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), name='conv4', activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(5, 5), name='conv5', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), name='pool3'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid', name='predictions'))

    return model
