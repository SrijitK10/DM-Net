from keras.preprocessing.image import ImageDataGenerator

def create_train_generator(train_path, img_size=(256, 256), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

def create_validation_generator(val_path, img_size=(256, 256), batch_size=32):
    val_datagen = ImageDataGenerator(rescale=1./255)
    return val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
