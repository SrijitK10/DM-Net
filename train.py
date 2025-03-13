import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from data import load_samples, data_generator
from model import LPQ_net

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Hyperparameters
batch_size = 32
learning_rate = 0.01
epochs = 30
num_classes = 2

def plot_and_save(history, save_dir='./training_plots'):
    """ Plot training metrics and save them as PNG files. """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract training history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_loss, 'b', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()

    print(f"Plots saved in {save_dir}")

def main():
    # Image size
    img_size = 256

    # Load and preprocess data
    train_data_path = '/Users/srijit/Documents/Projects Personal/FREQUENCY/DM-Net/csv/train.csv'
    validation_data_path = '/Users/srijit/Documents/Projects Personal/FREQUENCY/DM-Net/csv/validation.csv'

    train_samples = load_samples(train_data_path)
    validation_samples = load_samples(validation_data_path)

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")

    # Data generators
    train_generator = data_generator(train_samples, batch_size=batch_size, num_classes=num_classes)
    validation_generator = data_generator(validation_samples,  batch_size=batch_size, num_classes=num_classes)

    # Create and compile the model
    model = LPQ_net(in_shape=(img_size, img_size, 3), num_classes=num_classes)
    
    # Optimizer
    opti = Adagrad(learning_rate=learning_rate)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    print(model.summary())

    # Model saving path
    model_filepath = './models/DM_NET_DCT2.keras'

    # Callbacks
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

    # Adjusted steps per epoch
    steps_per_epoch = np.ceil(len(train_samples) / batch_size).astype(int)
    validation_steps = np.ceil(len(validation_samples) / batch_size).astype(int)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, reduce_lr]
    )

    # Plot and save training curves
    plot_and_save(history)

if __name__ == '__main__':
    main()
