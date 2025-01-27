import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

def preprocess_data(image, label):
    # Resize image to AlexNet standard input size
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [227, 227])  # AlexNet's original input size
    image = image / 255.0
    return image, label

def load_dataset(batch_size=32):
    # Load CIFAR-10
    (train_ds, val_ds), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    
    # Apply preprocessing and batching
    train_ds = train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def create_alexnet(num_classes=10):
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(227, 227, 3)),
        
        # First Convolutional Block
        layers.Conv2D(96, 11, strides=4, padding='valid', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        
        # Second Convolutional Block
        layers.Conv2D(256, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        
        # Third Convolutional Block
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # Fourth Convolutional Block
        layers.Conv2D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # Fifth Convolutional Block
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=2),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_alexnet(epochs=10, batch_size=32, learning_rate=0.001):
    # Load and prepare data
    train_ds, val_ds = load_dataset(batch_size)
    
    # Create model
    model = create_alexnet()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'alexnet_best.keras',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    return model, history

if __name__ == "__main__":
    model, history = train_alexnet()
