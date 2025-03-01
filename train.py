from keras.callbacks import ModelCheckpoint
from data_loader import create_train_generator, create_validation_generator
from model import LPQ_net
from utils import plot_metrics
from keras.optimizers import Adagrad

def train_model(train_path, val_path, img_size=(256, 256), batch_size=32, epochs=30):
    learning_rate = 0.001
    train_generator = create_train_generator(train_path, img_size, batch_size)
    validation_generator = create_validation_generator(val_path, img_size, batch_size)
    model = LPQ_net(input_shape=(256,256, 3))
    opti=Adagrad(learning_rate=learning_rate,decay=(learning_rate/epochs))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    
    plot_metrics(history)
    return model
