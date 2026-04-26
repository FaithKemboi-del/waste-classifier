import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib, os

# Paths
DATASET_DIR = r"C:\Users\Faith\OneDrive\Documents\waste-classifier\dataset"
MODEL_OUT    = r"C:\Users\Faith\OneDrive\Documents\waste-classifier\model"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 10
CLASSES      = ["metal", "organic", "paper", "plastic"]

# Data generators with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
val_data = val_gen.flow_from_directory(
    f"{DATASET_DIR}/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Build model — MobileNetV2 as base
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False  # freeze base layers

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(4, activation="softmax")  # 4 classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
os.makedirs(MODEL_OUT, exist_ok=True)
model.save(f"{MODEL_OUT}/waste_model.h5")
print("\nModel saved!")

# Print final accuracy
final_acc = history.history["val_accuracy"][-1]
print(f"Final validation accuracy: {final_acc:.2%}")