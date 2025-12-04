import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

BATCH_SIZE = 8
IMG_SIZE = (250, 250)
EPOCHS = 12

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    labels = "inferred",
    label_mode = "binary",
    color_mode = "rgb",
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE,
    shuffle = True,
    seed = 420
)

class_names = train_ds.class_names

train_ds = train_ds.prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    pathlib.Path("data"),
    labels = "inferred",
    label_mode = "binary",
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE,
    shuffle = True,
    seed = 42,
    validation_split = 0.2,
    subset = "validation"
)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape = (*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.summary()

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = EPOCHS
)

model.save("left_right_classifier.h5")