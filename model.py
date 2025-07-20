import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# === Dataset paths ===
train_dir = 'train'
val_dir = 'validation'

# === Image and batch settings ===
img_size = 224
batch_size = 32

# === Data preprocessing ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# === Transfer Learning Model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Compile model ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train the model ===
epochs = 5  # You can increase for better results
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# === Save the model and class names ===
# Create 'models/' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained model
model.save('models/fruit_classifier.h5')

# Save class names in the same order as during training
class_names = list(train_generator.class_indices.keys())
np.save('models/class_names.npy', class_names)

print("✅ Model training completed.")
print("✅ Model saved to: models/fruit_classifier.h5")
print("✅ Class names saved to: models/class_names.npy")
