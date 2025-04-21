import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ======================
# 1. DATA PREPARATION
# ======================
# Verify dataset paths exist
train_data_dir = "/content/dataset/facial_emotion_split/train"
validation_data_dir = "/content/dataset/facial_emotion_split/validation"

if not os.path.exists(train_data_dir) or not os.path.exists(validation_data_dir):
    raise FileNotFoundError("One or more dataset directories don't exist")

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1]  # Added brightness adjustment
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Batch size optimization
batch_size = 64  # Good for most GPUs

# Create data generators with validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42  # For reproducibility
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for proper validation
)

# Verify classes
if len(train_generator.class_indices) != 8:
    raise ValueError(f"Expected 8 classes, found {len(train_generator.class_indices)}")
print("Classes:", train_generator.class_indices)

# ======================
# 2. MODEL ARCHITECTURE
# ======================
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

# Enhanced optimizer configuration
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ======================
# 3. TRAINING SETUP
# ======================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',  
        patience=15,
        min_delta=0.001,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# ======================
# 4. MODEL TRAINING
# ======================
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=60,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // batch_size),
    callbacks=callbacks,
    verbose=1
)

# ======================
# 5. EVALUATION & VISUALIZATION
# ======================
# Save final model
model.save('emotion_model_final.keras')

# Evaluation
val_loss, val_acc = model.evaluate(validation_generator)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

# Enhanced visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accura
cy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()